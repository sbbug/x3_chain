set ff=UNIX
#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
set -ex
cd "$(dirname $0)" || exit
source ../../env.conf
source ../env.conf
if [ "$example_name" == "yolov3_onnx" ]; then
  sample_name=yolov3_onnx
fi
algo_name=${sample_name}
input_file_path=${val_data_path}
output_file_path='./preprocessed_data_yuv_lwir'
image_type=2
image_count=0 # means all
if [ "$TEST_CTX" == "SMOKE" ]; then
    image_count=100
fi
python3 ../../../02_runtime_src/4_simple_example/tools/send_tools/data_preprocess.py \
  --algo_name=${algo_name} \
  --input_file_path=${input_file_path}  \
  --output_file_path=${output_file_path} \
  --image_type=${image_type} \
  --image_count=${image_count}
