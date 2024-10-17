import torch
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_complex_scatters(y_train, y_pred_train, y_test, y_pred_test,
                          model_name='Unknown', trait_name='Unknown', save_dir=False):
    def prepare_data(target, output):
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy().reshape(-1)
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy().reshape(-1)

        valid_mask = ~np.isnan(target)
        target = target[valid_mask]
        output = output[valid_mask]

        r2 = r2_score(target, output)
        mape = np.mean(np.abs((target - output) / target)) * 100

        return target, output, r2, mape

    y_train, y_pred_train, r2_train, mape_train = prepare_data(y_train, y_pred_train)
    y_test, y_pred_test, r2_test, mape_test = prepare_data(y_test, y_pred_test)

    data_train = pd.DataFrame({
        'True': y_train,
        'Predicted': y_pred_train,
        'Data Set': 'Train'
    })
    data_test = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred_test,
        'Data Set': 'Test'
    })
    data = pd.concat([data_train, data_test])

    # Plot Settings
    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
    plt.figure(figsize=(8, 6), dpi=1200)
    g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)

    g.plot_joint(sns.scatterplot, alpha=0.5)  # 绘制中心的散点图
    # 回归线
    sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1', label='Train Regression Line')
    sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a', label='Test Regression Line')
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)  # 边缘柱状图

    # Text
    ax = g.ax_joint
    ax.text(0.6, 0.05, f'MAPE -- Train: {mape_train:.1f} %', transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='left')
    ax.text(0.85, 0.05, f'Test: {mape_test:.1f} %', transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='left')

    ax.text(0.6, 0.1, f'$R^2$      -- Train: {r2_train:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='left')
    ax.text(0.85, 0.1, f'Test: {r2_test:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='left')

    ax.text(0.5, 0.99, f'Trait = {trait_name}  Model = {model_name}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()],
            c="black", alpha=0.5, linestyle='--', label='x=y')
    ax.legend()

    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
    plt.show()


def plot_scatters(X, Y, title_str=''):
    # X: Ground Truth, Y: Prediction
    # 数据处理
    if torch.is_tensor(X) and torch.is_tensor(Y):
        assert X.shape == Y.shape, f"X and Y must have the same shape, but got X:{X.shape} and Y:{Y.shape}"
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()

    if isinstance(X, list) and isinstance(Y, list):
        assert len(X) == len(Y), f"X and Y must have the same length, but got X:{len(X)} and {len(Y)}"
        X = np.array(X)
        Y = np.array(Y)

    assert len(X) == len(Y), f"X and Y must have the same length, but got X:{len(X)}, Y:{len(Y)}"

    valid_mask = ~np.isnan(X)
    X = X[valid_mask]
    Y = Y[valid_mask]

    # 画散点
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X, Y, alpha=0.1)

    # 计算精度
    r2 = r2_score(Y, X)
    rmse = np.sqrt(mean_squared_error(Y, X))
    mape = np.mean(np.abs((Y - X) / Y)) * 100

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    ax.plot(X, slope * X + intercept, color='red', label='Linear regression line')

    # 注释
    plt.text(0.8, 0.4, f'points_num: {len(X)}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.8, 0.1, f'r_score: {r2:.3f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.8, 0.2, f'RMSE: {rmse:.3f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.8, 0.3, f'MAPE: {mape:.3f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)

    ax.set_aspect('equal', adjustable='box')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')

    # 显示设置
    plt.legend()
    plt.title(title_str)
    plt.tight_layout()

    plt.show()


def plot_importance(ips, wav='default', title_str=''):
    if wav == 'default':
        wav = np.arange(350, 2500, 1)

    assert len(ips) == len(wav), f'Expect same length of importance and wav, got ips:{len(ips)}|wav:{len(wav)}'

    # 画散点
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(wav, np.zeros(len(wav)), linestyle='--', color='black', linewidth=1)
    ax.fill_between(wav, ips, color='steelblue', alpha=0.5)

    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Importance')

    plt.title(title_str)
    plt.tight_layout()
    plt.show()

