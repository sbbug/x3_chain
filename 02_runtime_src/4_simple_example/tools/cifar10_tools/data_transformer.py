# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from easydict import EasyDict
from horizon_nn.data.transformer import *


def data_transformer():
    # means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    transformers = [
        ColorConvertTransformer('RGB', 'GRAY', 'CHW'),
    ]
    return transformers
