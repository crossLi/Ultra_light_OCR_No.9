# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F


def get_para_bias_attr(l2_decay, k, name):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_w_attr")
    bias_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_b_attr")
    return [weight_attr, bias_attr]


class GELU(nn.Layer):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + paddle.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * paddle.pow(x, 3))))


class SpatialAttention(nn.Layer):
    def __init__(self, attn_dim=768, head_num=12, dropout_ratio=0.2):
        super(SpatialAttention, self).__init__()
        self.head_num = head_num
        self.embedding = nn.Linear(attn_dim, attn_dim)
        self.q_linear = nn.Linear(attn_dim, attn_dim)
        self.k_linear = nn.Linear(attn_dim, attn_dim)
        self.v_linear = nn.Linear(attn_dim, attn_dim)
        # self.softmax = nn.Softmax(dim=-1)
        self.layernorm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.attn_dim = attn_dim

    def forward(self, x, spatial_matrix=None, mask=None):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        batch_size, seq_len, attn_dim = q.shape
        q = q.reshape((batch_size, self.head_num, 80, 8))
        k = k.reshape((batch_size, self.head_num, 8, 80))
        v = v.reshape((batch_size, self.head_num, 80, 8))
        alpha = paddle.matmul(q, k) / math.sqrt((self.attn_dim / self.head_num))
        if spatial_matrix is not None:
            alpha += spatial_matrix.sum(dim=-1)
        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(alpha, axis=-1)
        alpha = self.dropout(alpha)
        res = paddle.matmul(alpha, v).reshape((batch_size, 80, 96))
        x = x + self.dropout(res)
        x = self.layernorm(x)
        return x


class FeedForward(nn.Layer):
    def __init__(self, attn_dim=768, inner_dim=512, dropout_ratio=0.2):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(attn_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, attn_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.gelu = GELU()
        self.layernorm = nn.LayerNorm(attn_dim)

    def forward(self, x):
        res = self.linear1(x)
        res = self.gelu(res)
        res = self.linear2(res)
        x = x + self.dropout(res)
        x = self.layernorm(x)
        return x


# class FeedForward(nn.Module):
#     def __init__(self, hidden_size=768, intermediate_size=512, dropout=0.1):
#         super(FeedForward, self).__init__()
#         self.dense1 = nn.Linear(hidden_size, intermediate_size)
#         self.dense2 = nn.Linear(intermediate_size, hidden_size)
#         self.feedforward_act = GELU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, attention_x):
#         attention_x = self.dense1(attention_x)
#         attention_x = self.feedforward_act(attention_x)
#         attention_x = self.dense2(attention_x)
#         attention_x = self.dropout(attention_x)
#         return attention_x


class SpatialTransformer(nn.Layer):
    def __init__(self, attn_dim=768, head_num=12, dropout_ratio=0.2, inner_dim=512):
        super(SpatialTransformer, self).__init__()
        self.attention = SpatialAttention(attn_dim, head_num, dropout_ratio)
        self.feedforward = FeedForward(attn_dim, inner_dim, dropout_ratio)

    def forward(self, x, spatial_matrix=None, mask=None):
        x = self.attention(x, spatial_matrix, mask)
        x = self.feedforward(x)
        return x


class CTCHead(nn.Layer):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()

        # self.att = SpatialTransformer(attn_dim=in_channels, head_num=12, inner_dim=in_channels)
        self.linear1 = nn.Linear(in_channels, 128)
        self.emds = nn.Embedding(80, in_channels)

        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc')
        self.fc = nn.Linear(
            128,
            out_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='ctc_fc')
        self.out_channels = out_channels


    def forward(self, x, labels=None):
        # x = self.att(x)
        x = self.linear1(x)
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
        return predicts