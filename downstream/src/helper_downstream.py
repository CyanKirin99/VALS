
import logging
import sys

import torch

import upstream.src.models.encoders as encoders
from upstream.src.models.embedding import ManualEmbedding, AutomaticEmbedding
from downstream.src.models.regressor import Regressor
from upstream.src.utils.tensors import trunc_normal_
from upstream.src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model(
        device,
        tasks,
        model_name='encoder_base',
        pred_depth=4,
        pred_emb_dim=96,
        patch_size=30,
        preprocess='manual',
        proj_type='conv',
        pe_type='learnable',
        output_dims=None,
):
    encoder = encoders.__dict__[model_name](pe_type=pe_type)
    predictor = encoders.__dict__['encoder_predictor'](
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        pe_type=pe_type,
    )
    embedding = ManualEmbedding(encoder.embed_dim, patch_size, proj_type=proj_type) if preprocess == 'manual' \
        else AutomaticEmbedding(encoder.embed_dim, patch_size, proj_type=proj_type)
    regressor = Regressor(tasks=tasks, input_dim=encoder.embed_dim, output_dims=output_dims)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    for m in regressor.modules():
        init_weights(m)

    embedding.to(device)
    encoder.to(device)
    regressor.to(device)

    logger.info(regressor)

    return embedding, encoder, predictor, regressor


def load_checkpoint(
        device,
        r_path,
        embedding,
        encoder,
        predictor,
        regressor,
        opts=None,
        scalers=None,
        load_regressor=False,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch'] + 1

        # -- loading embedding
        pretrained_dict = checkpoint['embedding']
        msg = embedding.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained embedding with msg: {msg}')

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor with msg: {msg}')

        if load_regressor:
            try:
                pretrained_dict = checkpoint['regressor']
            except KeyError as e:
                print(f'No key named "regressor" found in checkpoint: {e}')
            msg = regressor.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained regressor from epoch {epoch} with msg: {msg}')
            # -- loading optimizer
            opts.load_state_dict(checkpoint['opt'])
            if scalers is not None:
                scalers.load_state_dict(checkpoint['scaler'])

        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return embedding, encoder, predictor, regressor, opts, scalers, epoch


def init_opt(
        regressor,
        iterations_per_epoch,
        start_lr,
        ref_lr,
        warmup,
        num_epochs,
        wd=1e-6,
        final_wd=1e-6,
        final_lr=0.0,
        use_bfloat16=False,
        ipe_scale=1.25
):
    tasks = regressor.tasks
    optimizers, scalers, schedulers, wd_schedulers = {}, {}, {}, {}

    for tk in tasks:
        param_groups = [
            {
                'params': (p for n, p in regressor.output[tk].named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in regressor.output[tk].named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }]

        logger.info('Using AdamW')
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup*iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=wd,
            final_wd=final_wd,
            T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
        scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

        optimizers[tk] = optimizer
        scalers[tk] = scaler
        schedulers[tk] = scheduler
        wd_schedulers[tk] = wd_scheduler

    return optimizers, scalers, schedulers, wd_schedulers
