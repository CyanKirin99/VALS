import numpy as np
import torch
import yaml
import os
import torch.nn.functional as F

from upstream.src.helper_upstream import load_checkpoint, init_model
from upstream.src.datasets.dataset import make_dataset
from upstream.src.utils.sampler import sample


def main(folder, tag):
    root = 'C:/file/Research/projects/TCAF/'

    fname = os.path.join(root, folder, f'params_{tag}.yaml')
    with open(fname, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    # -- META
    _GLOBAL_SEED = args['meta']['seed']
    model_name = args['meta']['model_name']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    patch_size = args['meta']['patch_size']
    proj_type = args['meta']['proj_type']
    learnable_pe = args['meta']['learnable_pe']
    manual = args['meta']['manual']

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu')

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    file_path = os.path.join(root, 'data/fresh.csv')

    # --MASK
    num_tgt_blk = args['mask']['num_tgt_blk']
    tgt_p_len = args['mask']['tgt_p_len']
    cxt_p_len = args['mask']['cxt_p_len']


    load_path = os.path.join(root, folder, f'{tag}-latest.pth.tar')

    embedding, encoder, predictor = init_model(
        device=device,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        patch_size=patch_size,
        proj_type=proj_type,
        manual=manual,
        learnable_pe=learnable_pe,
    )

    embedding, encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
        device=device,
        r_path=load_path,
        embedding=embedding,
        encoder=encoder,
        predictor=predictor,
        target_encoder=None,
        opt=None,
        scaler=None)

    # -- init data-loaders/samplers
    _, dataloader = make_dataset(file_path, batch_size, pin_mem, num_workers)
    ipe = len(dataloader)

    for itr, x in enumerate(dataloader):
        x = x.to(device)
        x, mask = embedding(x)

        tgt_indices, cxt_indices = sample(x, num_tgt_blk, tgt_p_len, cxt_p_len, mask)

        with torch.no_grad():
            # target forward
            all_rep = encoder(x, mask=mask)
            all_rep = F.layer_norm(all_rep, (all_rep.size(-1),))  # normalize over feature-dim
            all_rep = all_rep.repeat_interleave(num_tgt_blk, dim=0)
            # -- create targets (masked regions of h)
            tgt_indices_ = tgt_indices.unsqueeze(-1).expand(-1, -1, all_rep.size(-1))
            tgt_rep = torch.gather(all_rep, 1, tgt_indices_.to(all_rep.device))

            # context forward
            cxt_indices_ = cxt_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            cxt_rep = torch.gather(x, 1, cxt_indices_.to(x.device))
            cxt_rep = encoder(cxt_rep, indices=cxt_indices)


if __name__ == '__main__':
    folder = 'upstream/log/m_c_a_st_f'
    tag = 'm_c_a_st_f'

    main(folder, tag)
