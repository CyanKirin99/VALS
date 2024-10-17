import os
import numpy as np
import yaml

import torch

from eval.cnn.cnn_model import CNNModel
from downstream.src.datasets.dataset_refl_trait import make_dataset
from eval.utils.visualize import plot_scatters, plot_complex_scatters


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
    r_path = args['logging']['read_checkpoint']
    load_path = os.path.join(folder, r_path)

    model = CNNModel(in_channels, hid_channels, out_channels, kernel_size, stride, padding).to(device)

    dataset, train_loader, test_loader = make_dataset(spec_path, trait_path, tasks, split_ratio,
                                                      batch_size, pin_mem, num_workers)
    scaler_dict = dataset.scaler_dict
    median = scaler_dict[tk]['Median']
    IQR = scaler_dict[tk]['IQR']

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # eval
    def eval(loader):
        all_outputs, all_trait = [], []
        for itr, (x, trait) in enumerate(loader):
            x = x.to(device)
            t = trait[tk].to(device)

            with torch.no_grad():
                o = model(x)

            all_outputs.append(o)
            all_trait.append(t)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_trait = torch.cat(all_trait, dim=0)

        return all_outputs * IQR + median, all_trait * IQR + median

    train_outputs, train_trait = eval(train_loader)
    test_outputs, test_trait = eval(test_loader)

    # -- plot result
    for tk in tasks:
        y_train = train_trait
        y_pred_train = train_outputs
        y_test = test_trait
        y_pred_test = test_outputs
        plot_complex_scatters(y_train, y_pred_train, y_test, y_pred_test,
                              model_name='CNN', trait_name=tk)


if __name__ == '__main__':
    with open('configs_cnn.yaml', 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    main(args)


