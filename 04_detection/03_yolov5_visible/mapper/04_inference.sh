#!/bin/bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

set -e -v
cd $(dirname $0)

#IMAGE="../../../01_common/test_data/det_images/kite.jpg"
#MODEL="./model_output/yolov5_quantized_model.onnx"

#python3 inference.py --model=${MODEL} --image=${IMAGE}
python3 inference.py
