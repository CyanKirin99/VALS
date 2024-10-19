import argparse
import os
import logging
import pprint
import sys
import yaml

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from upstream.src.utils.loggings import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter,)
from downstream.src.datasets.dataset_refl_trait import make_dataset
from downstream.src.helper_downstream import (
    load_checkpoint,
    init_model,
    init_opt)
from downstream.src.models.upstream_model import CompleteUpstreamModel, IgnoreUpstreamModel


def main(args):
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
    pred_type = args['meta']['pred_type']  # downstream special
    notes = args['meta']['notes']

    patch_size = args['meta']['patch_size']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    output_dims = args['meta']['output_dims']  # downstream special

    load_processor = args['meta']['load_processor']
    r_file = args['meta']['read_checkpoint']
    freeze_encoder = args['meta']['freeze_encoder']  # downstream special


    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    spec_path = args['data']['spec_path']
    trait_path = args['data']['trait_path']
    tasks = args['data']['tasks']
    split_ratio = args['data']['split_ratio']

    # -- OPTIMIZATION
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    tag_ = f'{preprocess}_{proj_type}_{pe_type}_{model_size}_{dataset_name}_{pred_type}'
    folder = f'log/{tag_}/'
    tag = f'{tag_}_{notes}'
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
    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    columns = ([('%d', 'epoch'), ('%d', 'itr')] + [(f'%.6f', 'loss_'+tk) for tk in tasks] +
               [(f'%.2e', 'lr_'+tk) for tk in tasks] + [(f'%.2e', 'wd_'+tk) for tk in tasks])
    csv_logger = CSVLogger(log_file, *columns)

    model_name = f'encoder_{model_size}'
    embedding, encoder, predictor, processor = init_model(
        device=device,
        tasks=tasks,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        patch_size=patch_size,
        proj_type=proj_type,
        output_dims=output_dims,
        preprocess=preprocess,
        pe_type=pe_type,)

    # -- init data-loaders/samplers
    dataset, train_loader, test_loader = make_dataset(spec_path, trait_path, tasks, output_dims, split_ratio,
                                                      batch_size, pin_mem, num_workers)
    scaler_dict = dataset.scaler_dict
    ipe = len(train_loader)

    optimizers, scalers, schedulers, wd_schedulers = init_opt(
        processor=processor,
        iterations_per_epoch=ipe,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        wd=wd,
        final_wd=final_wd,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16,
        ipe_scale=ipe_scale
    )

    if freeze_encoder:
        for p in embedding.parameters():
            p.requires_grad = False
        for p in encoder.parameters():
            p.requires_grad = False

    start_epoch = 0
    # -- load training checkpoint
    embedding, encoder, predictor, processor, opt, scaler, epoch = load_checkpoint(
        device=device,
        r_path=load_path,
        embedding=embedding,
        encoder=encoder,
        predictor=predictor,
        processor=processor,
        opts=optimizers,
        scalers=scalers,
        load_processor=load_processor,)
    for _ in range(start_epoch*ipe):
        for tk in tasks:
            schedulers[tk].step()
            wd_schedulers[tk].step()

    # -- define upstream model
    if pred_type == 'complete':
        up_model = CompleteUpstreamModel(embedding, encoder, predictor).to(device)
    elif pred_type == 'ignore':
        up_model = IgnoreUpstreamModel(embedding, encoder).to(device)
    if freeze_encoder:
        up_model.eval()

    def save_checkpoint(epoch):
        save_dict = {
            'embedding': embedding.state_dict(),
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'processor': processor.state_dict(),
            'opt': optimizers,
            'scaler': None if scalers is None else scalers,
            'epoch': epoch,
            'batch_size': batch_size,
            'lr': lr
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logging.info(f'Epoch {epoch+1}')

        loss_meter = {tk: AverageMeter() for tk in tasks}
        time_meter = AverageMeter()

        for itr, (refl, trait) in enumerate(train_loader):
            refl = refl.to(device)
            lr_dict = {tk: 0.0 for tk in tasks}
            wd_dict = {tk: 0.0 for tk in tasks}
            grad_stats_dict = {tk: None for tk in tasks}

            def train_step():
                for tk in tasks:
                    optimizers[tk].zero_grad()
                    lr_dict[tk] = schedulers[tk].step()
                    wd_dict[tk] = wd_schedulers[tk].step()

                def loss_fn(o, t, tk):
                    output_dim = output_dims[tk]
                    if output_dim == 1:
                        return F.mse_loss(o, t.to(o.device))
                    else:
                        return F.cross_entropy(o, t.to(o.device).long())

                def forward_upstream(refl):
                    if freeze_encoder:
                        with torch.no_grad():
                            x, mask = up_model(refl)
                    else:
                        x, mask = up_model(refl)
                    return x, mask

                def forward_downstream(x, mask=None):
                    if pred_type == 'complete':
                        outputs = processor(x)
                    elif pred_type == 'ignore':
                        outputs = processor(x, mask=mask)

                    loss_dict = {tk: torch.tensor(0.0).to(device) for tk in tasks}
                    for tk in tasks:
                        valid_mask = ~torch.isnan(trait[tk])
                        o = outputs[tk][valid_mask]
                        t = trait[tk][valid_mask]
                        loss_dict[tk] = loss_fn(o, t, tk)
                    return outputs, loss_dict

                # -- Forward
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                    x, mask = forward_upstream(refl)
                    outputs, loss_dict = forward_downstream(x, mask)

                # -- Backward & step
                for tk in tasks:
                    loss = loss_dict[tk]
                    if not torch.isnan(loss):
                        loss.backward()
                        optimizers[tk].step()
                    grad_stats_dict[tk] = grad_logger(processor.output[tk].named_parameters())

                return loss_dict, lr_dict, wd_dict, grad_stats_dict

            (loss_dict, lr_dict, wd_dict, grad_stats_dict), etime = gpu_timer(train_step)
            for tk in tasks:
                loss_meter[tk].update(loss_dict[tk])
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                log_data = ([epoch+1, itr] + [loss_dict[tk] for tk in tasks] +
                            [lr_dict[tk] for tk in tasks] + [wd_dict[tk] for tk in tasks])
                csv_logger.log(*log_data)

                if (itr % log_freq == 0) or (any(torch.isnan(loss) for loss in loss_dict.values())):
                    loss_info = ' '.join([f'{tk}_loss: {loss_meter[tk].avg:.6f}\t' for tk in tasks])
                    logger.info('[%d %5d] '
                                '[%s] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_info,
                                   torch.cuda.max_memory_allocated() / 1024. ** 2,
                                   time_meter.avg))

                    for tk in tasks:
                        grad_stats = grad_stats_dict[tk]
                        if grad_stats is not None:
                            logger.info('[%d %5d] %s_grad_stats: [(%.2e %.2e)] [(%.2e, %.2e)]'
                                        % (epoch+1, itr, tk,
                                           grad_stats.first_ca_layer,
                                           grad_stats.last_ca_layer,
                                           grad_stats.min,
                                           grad_stats.max))

                for task, loss in loss_dict.items():
                    assert not torch.isnan(loss), f"Error: Loss for '{task}' is NaN."

            log_stats()

        epoch_loss_info = f'epoch_{epoch+1}\t' + ' '.join([f'{tk}_loss: {loss_meter[tk].avg:.6f}\t' for tk in tasks])
        logger.info(epoch_loss_info)
        save_checkpoint(epoch+1)


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
    parser.add_argument('--fname', type=str, default='configs_downstream.yaml')
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'])

    args = parser.parse_args()

    # num_gpus = len(args.devices)
    # mp.set_start_method('spawn')
    #
    # for rank in range(num_gpus):
    #     mp.Process(
    #         target=process_main,
    #         args=(rank, args.fname, num_gpus, args.devices)
    #     ).start()

    process_main(0, args.fname, 1, args.devices)
