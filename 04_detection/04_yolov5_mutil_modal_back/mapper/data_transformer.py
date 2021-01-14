# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import sys
sys.path.append("../../../01_common/python/data/")
from transformer import *


def data_transformer(input_shape=(672, 672)):
    transformers = [
        PadResizeTransformer(input_shape),
        TransposeTransformer((2, 0, 1)),
        ChannelSwapTransformer((2, 1, 0)),
        # ScaleTransformer(1 / 255),
    ]
    return transformers
