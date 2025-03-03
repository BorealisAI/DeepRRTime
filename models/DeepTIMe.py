# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2022-present, salesforce.com, inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
########################################################################################
# Code is based on the DeepTime (https://arxiv.org/abs/2207.06046) implementation
# from https://github.com/salesforce/DeepTime by Salesforce which is licensed under 
# BSD-3-Clause license. You may obtain a copy of the License at
# https://github.com/salesforce/DeepTime/blob/main/LICENSE.txt
########################################################################################

import random
import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat

from models.modules.inr import INR
from models.modules.regressors import RidgeRegressor


@gin.configurable()
def deeptime(datetime_feats: int, layer_size: int, inr_layers: int, n_fourier_feats: int, scales: float, cov_loss:str):
    return DeepTIMe(datetime_feats, layer_size, inr_layers, n_fourier_feats, scales, cov_loss)


class DeepTIMe(nn.Module):
    def __init__(self, datetime_feats: int, layer_size: int, inr_layers: int, n_fourier_feats: int, scales: float, cov_loss:str):
        super().__init__()
        self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales)
        self.adaptive_weights = RidgeRegressor()

        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.cov_loss = cov_loss
        self.mask_eval = 0

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)

        if y_time.shape[-1] != 0:
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)
            time_reprs = self.inr(coords)
        else:
            time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)

        aux_loss = 0 if self.cov_loss=='none' else self.cov_reg(time_reprs[:1])
        lookback_reprs = time_reprs[:, :-tgt_horizon_len]
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        w, b = self.adaptive_weights(lookback_reprs, x, mask=self.get_mask(x))
        preds = self.forecast(horizon_reprs, w, b)
        return preds, aux_loss

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')

    def cov_reg(self, reprs):
        B, T, D = reprs.shape # B, T, 256
        mu = reprs.mean(dim=1,keepdim=True)
        reprs = reprs - mu
        
        cov_reprs = (reprs.permute(0,2,1) @ reprs) / T
        lambda_cov = float(self.cov_loss[4:])
        
        return lambda_cov*torch.square(cov_reprs-torch.eye(D,device=reprs.device)[None]).mean() 
    
    def get_mask(self, x):
        mask = torch.ones_like(x[:,:,0:1])
        if not self.training:
            for i in range(x.shape[0]):
                indices = random.sample(range(x.shape[1]),int(self.mask_eval*x.shape[1]))
                mask[i,indices] *= 0
        return mask
