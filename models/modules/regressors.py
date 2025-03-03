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

from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] =0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None, mask=None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff, mask)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float, mask: Tensor) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)
        if not self.training:
            num_obs = int(mask.sum(dim=1)[0].item())
            assert (mask.sum(dim=1)==num_obs).all()
            X = X[mask.squeeze(-1)==1].view(batch_size,num_obs,n_dim+1)
            Y = Y[mask.squeeze(-1)==1].view(batch_size,num_obs,-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
