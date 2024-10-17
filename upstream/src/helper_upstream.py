import logging
import sys

import torch

import upstream.src.models.encoders as encoders
from upstream.src.models.embedding import ManualEmbedding, AutomaticEmbedding
from upstream.src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from upstream.src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
        device,
        r_path,
        encoder,
        predictor,
        target_encoder,
        embedding,
        opt,
        scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading embedding
        pretrained_dict = checkpoint['embedding']
        msg = embedding.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained embedding from epoch {epoch} with msg: {msg}')

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch + 1}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return embedding, encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
        device,
        model_name='encoder_base',
        pred_depth=4,
        pred_emb_dim=96,
        patch_size=30,
        preprocess='manual',
        proj_type='conv',
        pe_type='learnable',
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

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in embedding.modules():
        init_weights(m)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    embedding.to(device)
    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return embedding, encoder, predictor


def init_opt(
        encoder,
        predictor,
        embedding,
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
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in embedding.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in embedding.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
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
