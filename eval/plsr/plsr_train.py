import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.cross_decomposition import PLSRegression

from upstream.src.models.embedding import FilterResampler
from downstream.src.datasets.dataset_refl_trait import make_dataset
from eval.utils.visualize import plot_scatters, plot_complex_scatters


def main(args):
    # -- META
    _GLOBAL_SEED = args['meta']['seed']
    n_components = args['meta']['n_components']

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    spec_path = args['data']['spec_path']
    trait_path = args['data']['trait_path']
    tasks = args['data']['tasks']
    split_ratio = args['data']['split_ratio']

    save_dir = args['log']['save_dir']

    assert len(tasks) == 1, 'Only one task is supported'
    tk = tasks[0]

    # --

    dataset, train_loader, test_loader = make_dataset(spec_path, trait_path, tasks, split_ratio,
                                                      batch_size, pin_mem, num_workers)
    scaler_dict = dataset.scaler_dict
    median = scaler_dict[tk]['Median']
    IQR = scaler_dict[tk]['IQR']

    resampler = FilterResampler()

    def loader2numpy(loader):
        all_data = []
        all_trait = []

        for itr, (x, trait) in enumerate(loader):
            x_resampled = resampler(x)
            all_data.append(x_resampled.detach().numpy())

            trait_value = trait[tk]
            all_trait.append(trait_value.detach().numpy())

        # 拼接所有的batch
        X = np.concatenate(all_data, axis=0).astype(np.float64)
        Y = np.concatenate(all_trait, axis=0).astype(np.float64)

        # 筛除空值
        valid_mask = ~np.isnan(Y)
        X = X[valid_mask]
        Y = Y[valid_mask]

        return X, Y

    x_train, y_train = loader2numpy(train_loader)
    x_test, y_test = loader2numpy(test_loader)

    pls = PLSRegression(n_components=n_components)
    pls.fit(x_train, y_train)

    y_pred_train = pls.predict(x_train)
    y_pred_test = pls.predict(x_test)

    plot_complex_scatters(y_train * IQR + median, y_pred_train * IQR + median,
                          y_test * IQR + median, y_pred_test * IQR + median,
                          model_name='PLSR', trait_name='CHLa', save_dir=save_dir)


if __name__ == '__main__':
    with open('configs_plsr.yaml', 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    main(args)
