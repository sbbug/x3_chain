#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.


cd "$(dirname $0)" || exit

source ../../env.conf
source ../env.conf

algo_name=${sample_name}
input_file_path='./preprocessed_data_yuv/'
ip='localhost'
image_count=100
image_type=2

if [ $1 ]; then
  ip=$1
fi

if [ $2 ]; then
  image_count=$2
fi

python3 ../../../02_runtime_src/4_simple_example/tools/send_tools/send_image.py \
  --algo-name=${algo_name} \
  --input-file-path=${input_file_path} \
  --ip=${ip} \
  --image-count=${image_count} \
  --image-type=${image_type}
