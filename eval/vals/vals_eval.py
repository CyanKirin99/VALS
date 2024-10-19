import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from downstream.src.helper_downstream import init_model, load_checkpoint
from downstream.src.datasets.dataset_refl_trait import make_dataset
from eval.utils.denormalization import Denormalize
from eval.utils.visualize import plot_scatters, plot_complex_scatters
from downstream.src.models.upstream_model import IgnoreUpstreamModel, CompleteUpstreamModel

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)


def main(args):
    # -- META
    _GLOBAL_SEED = args['meta']['seed']

    preprocess = args['meta']['preprocess']
    proj_type = args['meta']['proj_type']
    model_size = args['meta']['model_size']
    pe_type = args['meta']['pe_type']
    pred_type = args['meta']['pred_type']  # downstream special

    patch_size = args['meta']['patch_size']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    output_dims = args['meta']['output_dims']

    load_processor = args['meta']['load_processor']
    r_file = args['meta']['read_checkpoint']

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
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    tasks = args['data']['tasks']
    split_ratio = args['data']['split_ratio']
    spec_path = args['data']['spec_path']
    trait_path = args['data']['trait_path']
    if isinstance(spec_path, dict):
        train_spec_path = spec_path['train']
        test_spec_path = spec_path['test']
        train_trait_path = trait_path['train']
        test_trait_path = trait_path['test']
        split_ratio = (1., 0.)

    # -- log/checkpointing paths
    folder = args['logging']['folder']
    load_path = os.path.join(folder, r_file)

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
        pe_type=pe_type)

    embedding, encoder, predictor, processor, _, _, _ = load_checkpoint(
        device=device,
        r_path=load_path,
        embedding=embedding,
        encoder=encoder,
        predictor=predictor,
        processor=processor,
        load_processor=load_processor)

    # -- define upstream model
    if pred_type == 'complete':
        up_model = CompleteUpstreamModel(embedding, encoder, predictor).to(device)
    elif pred_type == 'ignore':
        up_model = IgnoreUpstreamModel(embedding, encoder).to(device)
    up_model.eval()

    # -- init dataset
    if not isinstance(spec_path, dict):
        dataset, train_loader, test_loader = make_dataset(spec_path, trait_path, tasks, output_dims, split_ratio,
                                                          batch_size, pin_mem, num_workers)
    else:
        dataset, train_loader, _ = make_dataset(train_spec_path, train_trait_path, tasks, output_dims, split_ratio,
                                                batch_size, pin_mem, num_workers)
        dataset, test_loader, _ = make_dataset(test_spec_path, test_trait_path, tasks, output_dims, split_ratio,
                                               batch_size, pin_mem, num_workers)
    denorm = Denormalize(dataset.scaler_dict)

    # -- eval
    def eval(loader):
        all_outputs, all_trait = {}, {}
        for itr, (x, trait) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                x, mask = up_model(x)
                if pred_type == 'complete':
                    outputs = processor(x)
                elif pred_type == 'ignore':
                    outputs = processor(x, mask=mask)

            # -- save & transform
            for tk, o in outputs.items():
                if tk not in all_outputs:
                    all_outputs[tk] = []
                all_outputs[tk].append(o)
            for tk, t in trait.items():
                if tk not in all_trait:
                    all_trait[tk] = []
                all_trait[tk].append(t)

        for tk in all_outputs:
            all_outputs[tk] = torch.cat(all_outputs[tk], dim=0)  # 在第0维拼接
        for tk in all_trait:
            all_trait[tk] = torch.cat(all_trait[tk], dim=0)  # 在第0维拼接

        all_outputs, all_trait = denorm(all_outputs, all_trait, output_dims)
        return all_outputs, all_trait

    # -- 分类任务性能评估
    def eval_cls(y_train, y_pred_train, y_test, y_pred_test):
        def one_set(y_true, y_pred, set):
            y_true = y_true.detach().cpu().numpy()
            y_pred_classes = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred_classes)
            precision = precision_score(y_true, y_pred_classes, average='weighted')
            recall = recall_score(y_true, y_pred_classes, average='weighted')
            f1 = f1_score(y_true, y_pred_classes, average='weighted')

            results = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
            print(f"{set}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            cm = confusion_matrix(y_true, y_pred_classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()

            return results
        results_train = one_set(y_train, y_pred_train, 'Train')
        results_test = one_set(y_test, y_pred_test, 'Test')
        return results_train, results_test

    # -- 回归任务性能评估
    def eval_reg(y_train, y_pred_train, y_test, y_pred_test, tk):
        plot_complex_scatters(y_train, y_pred_train, y_test, y_pred_test,
                              model_name='VALS', trait_name=tk)

    train_outputs, train_trait = eval(train_loader)
    test_outputs, test_trait = eval(test_loader)

    # -- plot result
    for tk in tasks:
        y_train = train_trait[tk]
        y_pred_train = train_outputs[tk]
        y_test = test_trait[tk]
        y_pred_test = test_outputs[tk]

        if output_dims[tk] == 1:
            print(f"Evaluating regression task: {tk}")
            eval_reg(y_train, y_pred_train, y_test, y_pred_test, tk)
        else:
            print(f"Evaluating classification task: {tk}")
            results_train, results_test = eval_cls(y_train, y_pred_train, y_test, y_pred_test)


if __name__ == "__main__":
    with open('configs_vals.yaml', 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    main(args)
