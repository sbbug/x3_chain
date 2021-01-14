#!/usr/bin/env sh
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

cd "$(dirname $0)" || exit
. ./env.conf

runtime_model_file="./${sample_name}_hybrid_horizonrt.bin"
model_name=${sample_name}
perf_result_file="${sample_name}_perf.out"
# 1 for single core, 2 for dual core
core_num=1
# perf image number
image_number=10

rm -rf ./image_list.txt

for i in $(seq 1 ${image_number})
do
  echo ${board_test_image} >> image_list.txt
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./release/lib
./release/bin/example \
  --model_file=${runtime_model_file} \
  --model_name=${model_name} \
  --input_type=image \
  --input_config_string="{\"image_list_file\":\"image_list.txt\",\"width\":${input_width},\"height\":${input_height},\"data_type\":${input_type}}" \
  --output_type=raw \
  --output_config_string={\"output_file\":\"${perf_result_file}\"} \
  --core_num=${core_num} \
  --enable_post_process=true
