# -*- coding: utf-8 -*-
# @Author  : sjx_alo！！
# @FileName: MVCMGNet.py
# @Algorithm ：
# @Description:

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import stats
from torch.nn import Linear
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import ChebConv
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
device = 'cuda'

class LocalLayer(Module):
    # LocalLayer来自PGCN结构
    def __init__(self, in_features, out_features, bias=True):
        super(LocalLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lrelu = nn.LeakyReLU(0.1)
        self.bias = bias
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, lap, is_weight=True):
        if is_weight:
            weighted_feature = torch.einsum('b i j, j d -> b i d', input, self.weight)
            output = torch.einsum('i j, b j d -> b i d', lap, weighted_feature)+self.bias
        else:
            output = torch.einsum('i j, b j d -> b i d', lap, input)
        return output # (batch_size, 62, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features) }"

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        # Define trainable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] Correlation coefficient of graph attention (unnormalized)

        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GatingNetwork1(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(64, num_experts)
        )
        self.temperature = 1.0

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / self.temperature, dim=-1)


class GatingNetwork0(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(64, num_experts)
        )
        self.temperature = 1.0

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / self.temperature, dim=-1)



class MVCMGNet(nn.Module):
    def __init__(self):
        super(MVCMGNet, self).__init__()

        self.eeg_gat_list = nn.ModuleList([
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True),
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True),
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True),
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True)]
        )

        self.emg_gat_list = nn.ModuleList([
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True),
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True),
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True),
            GraphAttentionLayer(5000, 10,
                                dropout=0.3, alpha=0.1, concat=True)]
        )

        self.cmc_gcn_list = nn.ModuleList([
            LocalLayer(5000, 10, True),
            LocalLayer(5000, 10, True),
            LocalLayer(5000, 10, True),
            LocalLayer(5000, 10, True),
            LocalLayer(5000, 10, True)]
        )

        self.gating0 = GatingNetwork0(30, num_experts=3)
        self.gating1 = GatingNetwork1(40, num_experts=4)

        self.mlp0 = nn.Linear(40, 256)
        self.mlp1 = nn.Linear(256, 32)
        self.mlp2 = nn.Linear(32, 4)

        # common
        # self.layer_norm = nn.LayerNorm([30])
        self.bn = nn.BatchNorm1d(32)
        self.lrelu = nn.LeakyReLU(1e-4)
        self.dropout = nn.Dropout(0.5)
        # Output layer
        self.dense = Linear(128 * 2, 4)

        self.decoder =  Linear(120, 34)


    def forward(self, eeg, wpli_eeg, emg, wpli_emg, cmc, cmc_train):
        batch_index = torch.stack(
            [torch.tensor([i] * (30)) for i in range(eeg.shape[0])]).view(-1).cuda()
        if len(eeg.shape) == 4:
            batch_size, depth, channels, features = eeg.shape
        elif len(eeg.shape) == 3:
            batch_size, channels, features = eeg.shape
        else:
            bn_channels, features = eeg.shape
            eeg = eeg.reshape(torch.max(batch_index) + 1, -1, features)
            bn_channels, features = emg.shape
            emg = emg.reshape(torch.max(batch_index) + 1, -1, features)
            bn_channels, features = cmc.shape
            cmc = cmc.reshape(torch.max(batch_index) + 1, -1, features)

        cmc_cor = np.mean(cmc_train, axis=0)

        band_features = []
        for i in range(4):
            eeg_feature = self.eeg_gat_list[i](eeg[i], wpli_eeg[i])
            emg_feature = self.emg_gat_list[i](emg[i], wpli_emg[i])
            cmc_feature = self.cmc_gcn_list[i](cmc[i], cmc_cor[i])

            expert_outputs = torch.stack([eeg_feature, emg_feature, cmc_feature], dim=1)

            gate_weights_cmc = self.gating0(expert_outputs)  # [B, 2]
            weights_cmc = gate_weights_cmc.unsqueeze(-1)

            fused_feature = torch.sum(expert_outputs * weights_cmc, dim=1)

            band_features.append(fused_feature)

        expert_band_outputs = torch.stack(band_features, dim=1)

        gate_weights_cmc = self.gating1(expert_band_outputs)  # [B, 2]
        weights_cmc = gate_weights_cmc.unsqueeze(-1)

        out_feature = torch.sum(expert_band_outputs * weights_cmc, dim=1)

        x = F.relu(self.mlp0(out_feature))
        x = self.dropout(x)
        # x = self.bn(x)
        x = F.relu(self.mlp1(x))
        x = self.bn(x)
        # x = self.dropout(x)
        x = self.mlp2(x)

        return x

