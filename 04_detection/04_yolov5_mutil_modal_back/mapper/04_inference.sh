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

#IMAGE="/data/x3/x3_tc_1.1.17e/x3-toolchain-1.1.17e.tar/horizon_x3_tc_1.1.17e/samples/04_detection/02_yolov3_prune/test_img/0000068_02104_d_0000006.jpg"
#MODEL="./model_output/yolov3_quantized_model.onnx"

#python3 inference.py --model=${MODEL} --image=${IMAGE}
python3 inference.py