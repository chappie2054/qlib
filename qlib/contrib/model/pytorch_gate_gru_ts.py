# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import math
import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter


# class Gate(nn.Module):
#     def __init__(self, d_input, d_output, beta=5.0):
#         super().__init__()
#         self.trans = nn.Linear(d_input, d_output)
#         self.d_output = d_output
#         self.t = beta
#
#     def forward(self, gate_input):
#         output = self.trans(gate_input)
#         output = torch.softmax(output / self.t, dim=-1)
#         return self.d_output * output

# Gate V2.0
# class Gate(nn.Module):
#     def __init__(self, d_input, d_output, beta=5.0):
#         super().__init__()
#         # 编码层（可选：输入为IC、ICIR拼接）
#         self.embed = nn.Sequential(
#             nn.Linear(d_input, d_input),
#             nn.ReLU()
#         )
#         self.norm = nn.LayerNorm(d_input)
#         self.trans = nn.Sequential(
#             nn.Linear(d_input, d_input),
#             nn.ReLU(),
#             nn.Linear(d_input, d_output)
#         )
#         # self.t = beta  # 温度控制参数
#         self.log_t = nn.Parameter(torch.tensor(math.log(beta), dtype=torch.float32)) # 可训练温度
#
#     def forward(self, gate_input):
#         x = self.embed(gate_input)
#         x = self.norm(x)
#         logits = self.trans(x)
#         # weights = torch.softmax(logits / self.t, dim=-1)
#         weights = torch.softmax(logits / self.log_t.exp(), dim=-1)
#         return weights  # 直接返回 softmax 权重

# Gate V3.0  非线性
class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=5.0):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.ReLU(),
            nn.LayerNorm(d_input),
            nn.Linear(d_input, 2 * d_output)  # 输出为双通道：a 和 b
        )
        # 可训练温度（softmax 温度调节）
        self.log_t = nn.Parameter(torch.tensor(math.log(beta), dtype=torch.float32))

    def forward(self, gate_input):
        x = self.trans(gate_input)
        a, b = x.chunk(2, dim=-1)  # 切为两部分：a 为得分，b 为门控
        gated = a * torch.sigmoid(b)  # GLU 门控机制
        weights = torch.softmax(gated / self.log_t.exp(), dim=-1)
        return weights  # shape = [batch, d_output]


class GateGRU(Model):
    """GRU Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        window_size=3,
        beta=5,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        self.logger.info("GRU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "GRU parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\nwindow_size : {}"
            "\nbeta : {}"
            .format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
                window_size,
                beta,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GRU_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            # gate_input_dim=self.d_feat * window_size * 2,
            gate_input_dim=self.d_feat * window_size,
            beta=beta,
        )
        self.logger.info("model:\n{:}".format(self.GRU_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GRU_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GRU_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.GRU_model.train()

        for data, weight in data_loader:
            """
            data.shape: (N, T, F)
            N - number of batch size
            T - length of lookback time_step
            F - 6 factors + 2 * 6 * windows_size Rank_IC Rank_ICIR + 1 label
            """
            src = data[:, :, 0:self.d_feat].to(self.device) # factor_data -> N, T, D
            gate_input = data[:, -1, self.d_feat:-1].to(self.device).float() # gate_input_data ->  N, T, D
            label = data[:, -1, -1].to(self.device)

            pred = self.GRU_model(src.float(), gate_input)
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.GRU_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            src = data[:, :, 0:self.d_feat].to(self.device)  # factor_data -> N, T, D
            gate_input = data[:, -1, self.d_feat:-1].to(self.device).float()  # gate_input_data ->  N, T, D
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(src.float(), gate_input)
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            src = data[:, :, 0:self.d_feat].to(self.device)  # factor_data -> N, T, D
            gate_input = data[:, -1, self.d_feat:-1].to(self.device).float()  # gate_input_data ->  N, T, D

            with torch.no_grad():
                pred = self.GRU_model(src.float(), gate_input).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, gate_input_dim=None, beta=1.0):
        super().__init__()
        self.gate = Gate(gate_input_dim, d_feat, beta=beta)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x, gate_input):
        # x: (N, T, D); gate_input: (N, D2)
        gate_weight = self.gate(gate_input).unsqueeze(1)  # (N, 1, D)
        x = x * gate_weight                              # gated input
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
