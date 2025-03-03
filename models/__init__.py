# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see https://github.com/salesforce/DeepTime/blob/main/LICENSE.txt

from typing import Union

import torch

from .DeepTIMe import deeptime


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'deeptime':
        model = deeptime(datetime_feats=kwargs['datetime_feats'])
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
