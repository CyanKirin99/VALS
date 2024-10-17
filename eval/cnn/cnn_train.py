import os
import numpy as np
import yaml

import torch
import torch.nn.functional as F

from eval.cnn.cnn_model import CNNModel
from downstream.src.datasets.dataset_refl_trait import make_dataset
from upstream.src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)


def main(args):
    # -- META
    _GLOBAL_SEED = args['meta']['seed']
    in_channels = args['meta']['in_channels']
    hid_channels = args['meta']['hid_channels']
    out_channels = args['meta']['out_channels']
    kernel_size = args['meta']['kernel_size']
    stride = args['meta']['stride']
    padding = args['meta']['padding']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    # -- DATA
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    pin_mem = args['data']['pin_mem']
    spec_path = args['data']['spec_path']
    trait_path = args['data']['trait_path']
    split_ratio = args['data']['split_ratio']
    tasks = args['data']['tasks']
    assert len(tasks) == 1, 'Only one task is supported'
    tk = tasks[0]

    # -- LOGGING
    folder = args['logging']['folder']
    checkpoint_freq = args['logging']['checkpoint_freq']
    log_file = os.path.join(folder, f'{tk}.csv')
    save_path = os.path.join(folder, f'{tk}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tk}-latest.pth.tar')

    # -- OPTIMIZATION
    epochs = args['optimization']['epochs']
    final_lr = args['optimization']['final_lr']
    final_weight_decay = args['optimization']['final_weight_decay']
    ipe_scale = args['optimization']['ipe_scale']
    lr = args['optimization']['lr']
    start_lr = args['optimization']['start_lr']
    warmup = args['optimization']['warmup']
    weight_decay = args['optimization']['weight_decay']
    use_bfloat16 = args['optimization']['use_bfloat16']

    model = CNNModel(in_channels, hid_channels, out_channels, kernel_size, stride, padding).to(device)

    dataset, train_loader, test_loader = make_dataset(spec_path, trait_path, tasks, split_ratio,
                                                      batch_size, pin_mem, num_workers)
    ipe = len(train_loader)

    def init_opt(
            model,
            iterations_per_epoch,
            start_lr,
            ref_lr,
            warmup,
            num_epochs,
            wd=1e-6,
            final_wd=1e-6,
            final_lr=0.0,
            use_bfloat16=False,
            ipe_scale=1.25):

        param_groups = [
            {
                'params': (p for n, p in model.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in model.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }]

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=wd,
            final_wd=final_wd,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
        scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

        return optimizer, scaler, scheduler, wd_scheduler

    optimizer, scaler, scheduler, wd_scheduler = init_opt(model, ipe, start_lr, lr, warmup, epochs, weight_decay,
                                                          final_weight_decay, final_lr, use_bfloat16, ipe_scale)

    def compute_loss(o, t):
        valid_mask = ~torch.isnan(t)
        o = o[valid_mask]
        t = t[valid_mask]
        loss = F.mse_loss(o, t)
        return loss

    def save_checkpoint(epoch):
        save_dict = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    for epoch in range(epochs):
        for itr, (x, trait) in enumerate(train_loader):
            x = x.to(device)
            t = trait[tk].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                o = model(x)
                loss = compute_loss(o, t)

            optimizer.zero_grad()
            if use_bfloat16:
                scaler.scale(loss).backward()
                optimizer.step()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            wd_scheduler.step()

            print(f'Epoch: {epoch+1}\tIter:{itr+1}/{ipe}\tLoss: {loss.item():.5f}')

        save_checkpoint(epoch)


if __name__ == '__main__':
    with open('configs_cnn.yaml', 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    main(args)
