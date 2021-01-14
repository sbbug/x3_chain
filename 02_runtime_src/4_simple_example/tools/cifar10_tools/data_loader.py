# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from easydict import EasyDict
from horizon_nn.data.data_loader import *
from horizon_nn.data.transformer import *
from loader import CifarLoader


def data_transformer():
    # means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    transformers = [
        # ScaleTransformer( 1/ 255),
        # MeanTransformer(means),
        # ScaleTransformer(2),
        ColorConvertTransformer('RGB', 'GRAY', 'CHW'),
    ]
    return transformers


def CifarDataLoader(transformers,
                    cifar_path,
                    include_label=False,
                    max_len=0,
                    batch_size=1):
    loader = CifarLoader(
        cifar_path, include_label=include_label, max_len=max_len)
    return DataLoader(loader, transformers=transformers, batch_size=batch_size)


def dataset_loader(
        cifar_path=None,
        max_len=10000,
        include_label=False,
        batch_size=1,
):
    transformers = data_transformer()
    loader = CifarDataLoader(
        transformers,
        cifar_path=cifar_path,
        batch_size=batch_size,
        include_label=include_label,
        max_len=max_len)
    return loader
