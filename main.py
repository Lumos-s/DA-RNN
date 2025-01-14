import typing
from typing import Tuple
import json
import os

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from modules import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device
import random

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")
writer = SummaryWriter()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 这里为防止数据泄露，应当仅使用训练集的数据来标准化数据，由于训练集的划分不在此函数中，所以应当将训练集、测试集和验证集的数据分开处理
def preprocess_data(
    dat, col_names, train_size, scaler=StandardScaler()
) -> Tuple[TrainData, StandardScaler]:  # 数据预处理
    """
    数据预处理
    :param dat: 数据
    :param col_names: 列名
    :param train_size: 训练集大小
    :param val_size: 验证集大小
    :param scaler: 标准化器
    :return: 训练数据、验证数据、测试数据和标准化器
    """
    scale = scaler.fit(dat.head(int(train_size)))  # 标准化
    proc_dat = scale.transform(dat)  # 数据标准化

    mask = np.ones(
        proc_dat.shape[1], dtype=bool
    )  # 掩码，全为True，长度为列数，即特征数
    dat_cols = list(dat.columns)  # 列名
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False  # 将目标列的掩码设置为False

    feats = proc_dat[:, mask]  # 特征，去掉目标列
    targs = proc_dat[:, ~mask]  # 目标列

    return TrainData(feats, targs), scale


def da_rnn(
    train_data: TrainData,
    n_targs: int,
    train_size: int,
    val_size: int,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
    T=10,
    learning_rate=0.001,
    batch_size=128,
):
    """
    构建模型
    :param train_data: 训练数据
    :param n_targs: 目标列数
    :param encoder_hidden_size: 编码器隐藏层大小
    :param decoder_hidden_size: 解码器隐藏层大小
    :param T: 时间步长
    :param learning_rate: 学习率
    :param batch_size: 批大小
    :return: 训练配置和模型
    """
    train_cfg = TrainConfig(
        T,
        int(train_size),
        int(val_size),
        batch_size,
        nn.MSELoss(),
    )  # 训练配置
    logger.info(f"Training size: {train_cfg.train_size:d}.")  # 输出训练大小
    logger.info(f"Validation size: {train_cfg.val_size:d}.")  # 输出验证大小

    enc_kwargs = {
        "input_size": train_data.feats.shape[1],
        "hidden_size": encoder_hidden_size,
        "T": T,
    }  # 编码器参数，输入大小、隐藏层大小、时间步长

    encoder = Encoder(**enc_kwargs).to(device)  # 编码器
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)  # 保存编码器参数

    dec_kwargs = {
        "encoder_hidden_size": encoder_hidden_size,
        "decoder_hidden_size": decoder_hidden_size,
        "T": T,
        "out_feats": n_targs,
    }  # 解码器参数，编码器隐藏层大小、解码器隐藏层大小、时间步长、输出特征数
    decoder = Decoder(**dec_kwargs).to(device)  # 解码器
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)  # 保存解码器参数

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad], lr=learning_rate
    )  # 编码器优化器，Adam
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad], lr=learning_rate
    )  # 解码器优化器，Adam
    da_rnn_net = DaRnnNet(
        encoder, decoder, encoder_optimizer, decoder_optimizer
    )  # DaRnn网络

    return train_cfg, da_rnn_net


# 在预测结果上应用反向变换
def inverse_transform_predictions(predictions, scaler, col_names, dat):
    # 创建一个全为0的数组，形状与原始数据相同
    inverse_data = np.zeros((predictions.shape[0], dat.shape[1]))

    # 将预测结果放回到对应的列
    for i, col_name in enumerate(col_names):
        col_idx = list(dat.columns).index(col_name)
        inverse_data[:, col_idx] = predictions[:, i]

    # 进行反向变换
    inverse_data = scaler.inverse_transform(inverse_data)

    # 只返回反向变换后的预测结果部分
    return inverse_data[
        :, [list(dat.columns).index(col_name) for col_name in col_names]
    ]


def train(
    net: DaRnnNet,
    train_data: TrainData,
    t_cfg: TrainConfig,
    n_epochs=10,
    save_plots=False,
    scaler=None,
    targ_cols=None,
    raw_data=None,
):
    """
    训练模型
    :param net: DaRnn网络
    :param train_data: 训练数据
    :param t_cfg: 训练配置
    :param n_epochs: 迭代次数
    :param save_plots: 是否保存图片
    :return: 迭代损失和每轮损失
    """
    iter_per_epoch = int(
        np.ceil(t_cfg.train_size * 1.0 / t_cfg.batch_size)
    )  # 每轮迭代次数
    iter_losses = np.zeros(n_epochs * iter_per_epoch)  # 迭代损失
    iter_mae = np.zeros(n_epochs * iter_per_epoch)
    iter_mape = np.zeros(n_epochs * iter_per_epoch)
    iter_rmse = np.zeros(n_epochs * iter_per_epoch)

    epoch_losses = np.zeros(n_epochs)  # 每轮损失
    epoch_maes = np.zeros(n_epochs)
    epoch_mapes = np.zeros(n_epochs)
    epoch_rmses = np.zeros(n_epochs)

    logger.info(
        f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}."
    )  # 输出每轮迭代次数

    n_iter = 0  # 迭代次数初始化

    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)  # 随机排列
        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i : (t_i + t_cfg.batch_size)]  # 批索引
            feats, y_history, y_target = prep_train_data(
                batch_idx, t_cfg, train_data
            )  # 准备训练数据

            loss, mae, mape, rmse = train_iteration(
                net,
                t_cfg.loss_func,
                feats,
                y_history,
                y_target,
                scaler=scaler,
                targat_col=targ_cols,
                raw_data=raw_data,
            )  # 训练迭代
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = (
                loss  # 记录损失
            )

            iter_mae[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = mae
            iter_mape[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = mape
            iter_rmse[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = rmse

            n_iter += 1

            adjust_learning_rate(net, n_iter)  # 调整学习率

        epoch_losses[e_i] = np.mean(
            iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)]
        )

        epoch_maes[e_i] = np.mean(
            iter_mae[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)]
        )
        epoch_mapes[e_i] = np.mean(
            iter_mape[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)]
        )
        epoch_rmses[e_i] = np.sqrt(
            np.mean(
                np.square(
                    iter_rmse[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)]
                )
            )
        )

        # 在训练过程中记录标量值
        writer.add_scalar("train_loss", epoch_losses[e_i], e_i + 1)
        writer.add_scalar("train_mae", epoch_maes[e_i], e_i + 1)
        writer.add_scalar("train_mape", epoch_mapes[e_i], e_i + 1)
        writer.add_scalar("train_rmse", epoch_rmses[e_i], e_i + 1)

        y_test_pred = predict(
            net,
            train_data,
            t_cfg.train_size,
            t_cfg.val_size,
            t_cfg.batch_size,
            t_cfg.T,
            on_train=False,
            on_test=False,
        )

        val_targets = train_data.targs[
            t_cfg.train_size : t_cfg.train_size + t_cfg.val_size
        ]
        val_loss = y_test_pred - val_targets  # 计算验证损失，计算公式为预测值-真实值
        val_targets_inverse = inverse_transform_predictions(
            val_targets,
            scaler,
            targ_cols,
            raw_data,
        )
        y_test_pred_inverse = inverse_transform_predictions(
            y_test_pred,
            scaler,
            targ_cols,
            raw_data,
        )

        val_mae = mean_absolute_error(val_targets_inverse, y_test_pred_inverse)
        val_mape = mean_absolute_percentage_error(
            val_targets_inverse, y_test_pred_inverse
        )
        val_rmse = root_mean_squared_error(val_targets_inverse, y_test_pred_inverse)

        logger.info(
            f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}."
        )

        writer.add_scalar("val_loss", np.mean(np.abs(val_loss)), e_i + 1)
        writer.add_scalar("val_mae", val_mae, e_i + 1)
        writer.add_scalar("val_mape", val_mape, e_i + 1)
        writer.add_scalar("val_rmse", val_rmse, e_i + 1)

        y_train_pred = predict(
            net,
            train_data,
            t_cfg.train_size,
            t_cfg.val_size,
            t_cfg.batch_size,
            t_cfg.T,
            on_train=True,
            on_test=False,
        )

        # TODO: 这里预测值的起始时间步对应于真实值的哪一个时间步？对于预测的评估是否存在时间步错位？
        if e_i % 10 == 0:
            plt.figure()
            plt.plot(
                range(t_cfg.T, t_cfg.T + len(y_train_pred) + len(y_test_pred)),
                inverse_transform_predictions(
                    train_data.targs,
                    scaler,
                    targ_cols,
                    raw_data,
                )[: (len(y_train_pred) + len(y_test_pred))],
                label="True",
            )
            plt.plot(
                range(t_cfg.T, t_cfg.T + len(y_train_pred)),
                inverse_transform_predictions(
                    y_train_pred,
                    scaler,
                    targ_cols,
                    raw_data,
                ),
                label="Predicted - Train",
            )
            plt.plot(
                range(
                    t_cfg.T + len(y_train_pred),
                    t_cfg.T + len(y_train_pred) + len(y_test_pred),
                ),
                inverse_transform_predictions(
                    y_test_pred,
                    scaler,
                    targ_cols,
                    raw_data,
                ),
                label="Predicted - Test",
            )
            plt.legend(loc="upper left")
            utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T)
        feats[b_i, :, :] = train_data.feats[b_slc, :]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(
            net.enc_opt.param_groups, net.dec_opt.param_groups
        ):
            enc_params["lr"] = enc_params["lr"] * 0.9
            dec_params["lr"] = dec_params["lr"] * 0.9


def train_iteration(
    t_net: DaRnnNet,
    loss_func: typing.Callable,
    X,
    y_history,
    y_target,
    scaler=None,
    targat_col=None,
    raw_data=None,
):
    """
    训练迭代
    :param t_net: DaRnn网络
    :param loss_func: 损失函数
    :param X: 特征
    :param y_history: 历史目标
    :param y_target: 目标
    :return: 损失
    """
    t_net.enc_opt.zero_grad()  # 编码器优化器梯度清零
    t_net.dec_opt.zero_grad()  # 解码器优化器梯度清零

    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))  # 编码器
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))  # 解码器
    # logger.debug(f"Prediction: {y_pred}. Target: {y_target}.")  # 输出预测和目标

    y_true = numpy_to_tvar(y_target)  # 目标
    loss = loss_func(y_pred, y_true)  # 损失

    y_true_inverse = inverse_transform_predictions(
        y_true.cpu().detach().numpy(),
        scaler,
        targat_col,
        raw_data,
    )
    y_pred_inverse = inverse_transform_predictions(
        y_pred.cpu().detach().numpy(),
        scaler,
        targat_col,
        raw_data,
    )

    # 计算反向转换后的MAE、MAPE、RMSE
    mae = mean_absolute_error(y_true_inverse, y_pred_inverse)
    mape = mean_absolute_percentage_error(y_true_inverse, y_pred_inverse)
    rmse = root_mean_squared_error(y_true_inverse, y_pred_inverse)

    loss.backward()  # 反向传播

    t_net.enc_opt.step()  # 编码器优化器
    t_net.dec_opt.step()  # 解码器优化器

    return loss.item(), mae, mape, rmse


def predict(
    t_net: DaRnnNet,
    t_dat: TrainData,
    train_size: int,
    val_size: int,
    batch_size: int,
    T: int,
    on_train=False,
    on_test=False,
):
    out_size = t_dat.targs.shape[1]  # 输出大小
    test_size = t_dat.feats.shape[0] - train_size - val_size  # 测试大小
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))  # 预测,初始化为0
    elif on_test:
        y_pred = np.zeros((test_size, out_size))  # 预测,初始化为0
    else:
        y_pred = np.zeros((val_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]  # 批索引
        b_len = len(batch_idx)  # 批大小
        X = np.zeros((b_len, T, t_dat.feats.shape[1]))  # 特征 ,初始化为0
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idxx = range(b_idx, b_idx + T)
                idxy = range(b_idx, b_idx + T - 1)
            elif on_test:
                idxx = range(
                    b_idx + train_size + val_size - T, b_idx + train_size + val_size
                )
                idxy = range(
                    b_idx + train_size + val_size - T, b_idx + train_size + val_size - 1
                )
            else:
                idxx = range(b_idx + train_size - T, b_idx + train_size)
                idxy = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idxx, :]
            y_history[b_i, :] = t_dat.targs[idxy]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred


def main():
    setup_seed(42)
    train_size = 35100
    val_size = 2730

    save_plots = True  # 保存图片
    debug = False  # 调试模式

    raw_data = pd.read_csv(
        os.path.join("data", "nasdaq100_padding.csv"), nrows=100 if debug else None
    )  # 读取数据

    logger.info(
        f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}."
    )  # 输出数据的形状和缺失值

    targ_cols = ("NDX",)  # 目标列

    data, scaler = preprocess_data(
        raw_data,
        targ_cols,
        train_size,
        MinMaxScaler(),
    )  # 数据预处理

    da_rnn_kwargs = {"batch_size": 128, "T": 10}  # 参数，批大小和时间步长

    config, model = da_rnn(
        data,
        n_targs=len(targ_cols),
        train_size=train_size,
        val_size=val_size,
        learning_rate=0.001,
        **da_rnn_kwargs,
    )  # 构建模型

    iter_loss, epoch_loss = train(
        model,
        data,
        config,
        n_epochs=100,
        save_plots=save_plots,
        scaler=scaler,
        targ_cols=targ_cols,
        raw_data=raw_data,
    )  # 训练模型

    final_y_pred = predict(
        model,
        data,
        config.train_size,
        config.val_size,
        config.batch_size,
        config.T,
        on_train=False,
        on_test=True,
    )
    test_size = data.feats.shape[0] - config.train_size - config.val_size
    test_targets = data.targs[config.train_size + config.val_size :]

    test_targets_inverse = inverse_transform_predictions(
        test_targets, scaler, targ_cols, raw_data
    )
    final_y_pred_inverse = inverse_transform_predictions(
        final_y_pred, scaler, targ_cols, raw_data
    )

    pred_mae = mean_absolute_error(test_targets_inverse, final_y_pred_inverse)
    pred_mape = mean_absolute_percentage_error(
        test_targets_inverse, final_y_pred_inverse
    )
    pred_rmse = root_mean_squared_error(test_targets_inverse, final_y_pred_inverse)

    logger.info(
        f"Prediction on test set: MAE {pred_mae:.9f}, MAPE {pred_mape:.9f}, RMSE {pred_rmse:.9f}."
    )

    plt.figure()
    plt.semilogy(range(len(iter_loss)), iter_loss)
    utils.save_or_show_plot("iter_loss.png", save_plots)

    plt.figure()
    plt.semilogy(range(len(epoch_loss)), epoch_loss)
    utils.save_or_show_plot("epoch_loss.png", save_plots)

    plt.figure()
    plt.plot(final_y_pred, label="Predicted")
    plt.plot(data.targs[config.train_size + config.val_size :], label="True")
    plt.legend(loc="upper left")
    utils.save_or_show_plot("final_predicted.png", save_plots)

    with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
        json.dump(da_rnn_kwargs, fi, indent=4)

    joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
    torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
    torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))


if __name__ == "__main__":
    main()
