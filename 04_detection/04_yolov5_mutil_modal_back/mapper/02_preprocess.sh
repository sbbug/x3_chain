#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

set -e -v
cd $(dirname $0) || exit

python3 ../../data_preprocess.py \
  --src_dir /data/x3/x3_tc_1.1.17e/x3-toolchain-1.1.17e.tar/horizon_x3_tc_1.1.17e/samples/04_detection/04_yolov5_mutil_modal_back/kaist/visible \
  --dst_dir ./calibration_data_rgbp_visible

python3 ../../data_preprocess.py \
  --src_dir /data/x3/x3_tc_1.1.17e/x3-toolchain-1.1.17e.tar/horizon_x3_tc_1.1.17e/samples/04_detection/04_yolov5_mutil_modal_back/kaist/lwir \
  --dst_dir ./calibration_data_rgbp_lwir