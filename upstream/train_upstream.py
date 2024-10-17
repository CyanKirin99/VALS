import os
import copy
import logging
import sys
import yaml
import pprint
import argparse

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from upstream.src.utils.sampler import sample
from upstream.src.utils.loggings import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from upstream.src.utils.tensors import repeat_interleave_batch
from upstream.src.datasets.dataset import make_dataset

from upstream.src.helper_upstream import (
    load_checkpoint,
    init_model,
    init_opt)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    _GLOBAL_SEED = args['meta']['seed']
    use_bfloat16 = args['meta']['use_bfloat16']

    preprocess = args['meta']['preprocess']
    proj_type = args['meta']['proj_type']
    model_size = args['meta']['model_size']
    pe_type = args['meta']['pe_type']
    dataset_name = args['meta']['dataset_name']

    patch_size = args['meta']['patch_size']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    file_path = args['data']['file_path']

    # --MASK
    num_tgt_blk = args['mask']['num_tgt_blk']
    tgt_p_len = args['mask']['tgt_p_len']
    cxt_p_len = args['mask']['cxt_p_len']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    tag = f'{preprocess}_{proj_type}_{pe_type}_{model_size}_{dataset_name}'
    folder = f'log/{tag}/'
    log_timings = args['logging']['log_timings']
    log_freq = args['logging']['log_freq']
    checkpoint_freq = args['logging']['checkpoint_freq']

    dump = os.path.join(folder, f'params_{tag}.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.6f', 'loss'),
                           ('%d', 'time (ms)'))

    # -- init model
    model_name = f'encoder_{model_size}'
    embedding, encoder, predictor = init_model(
        device=device,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        patch_size=patch_size,
        proj_type=proj_type,
        preprocess=preprocess,
        pe_type=pe_type,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- init data-loaders/samplers
    _, dataloader = make_dataset(file_path, batch_size, pin_mem, num_workers)
    ipe = len(dataloader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        embedding=embedding,
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        embedding, encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            embedding=embedding,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    def save_checkpoint(epoch):
        save_dict = {
            'embedding': embedding.state_dict(),
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, x in enumerate(dataloader):
            x = x.to(device)
            x, mask = embedding(x)

            # tgt_indices: tensor (batch_size*num_tgt_blk, tgt_p_len)
            # cxt_indices: tensor (batch_size, cxt_p_len)
            tgt_indices, cxt_indices = sample(x, num_tgt_blk, tgt_p_len, cxt_p_len, mask)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(x, tgt_indices):
                    with torch.no_grad():
                        all_rep = target_encoder(x, mask=mask)
                        # all_rep = F.layer_norm(all_rep, (all_rep.size(-1),))  # normalize over feature-dim
                        all_rep = all_rep.repeat_interleave(num_tgt_blk, dim=0)
                        # -- create targets (masked regions of h)
                        tgt_indices_ = tgt_indices.unsqueeze(-1).expand(-1, -1, all_rep.size(-1))
                        tgt_rep = torch.gather(all_rep, 1, tgt_indices_.to(all_rep.device))
                    return tgt_rep, tgt_indices_

                def forward_context(x, cxt_indices):
                    cxt_indices_ = cxt_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
                    cxt_rep = torch.gather(x, 1, cxt_indices_.to(x.device))
                    cxt_rep = encoder(cxt_rep, indices=cxt_indices)
                    return cxt_rep, cxt_indices_

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                    tgt_rep, tgt_indices_ = forward_target(x, tgt_indices)
                    cxt_rep, cxt_indices_ = forward_context(x, cxt_indices)
                    tgt_shp = tgt_rep.shape
                    pred_rep = predictor(cxt_rep, tgt_shp, cxt_indices, tgt_indices, num_tgt_blk)
                    loss = loss_fn(pred_rep, tgt_rep)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.6f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.6f' % loss_meter.avg)
        save_checkpoint(epoch)


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    logger.info(f'Running... )')
    main(args=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='configs_upstream.yaml')
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'])

    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()

    # process_main(0, args.fname, 1, args.devices)
