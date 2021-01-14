#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

export HORIZON_RUN_CONTEXT=SIM
set -ex
cd "$(dirname $0)" || exit

source ../../env.conf
source ../env.conf

runtime_model_file="../mapper/model_output/${sample_name}_hybrid_horizonrt.bin"
model_name=${sample_name}
eval_result_file="${sample_name}_eval.out"
input_file_path='./preprocessed_data_yuv'
image_count=50000
if [ "$TEST_CTX" == "SMOKE" ]; then
  image_count=10
fi

find ${input_file_path} -name "*.bin" | head -n ${image_count} >eval_yuv_list.txt
# parallel_num=$(cat /proc/cpuinfo |grep processor|sed '$!d'|awk '{print $3}')
if [ "$PARALLEL_PROCESS_NUM" ]; then
  parallel_num=$PARALLEL_PROCESS_NUM
else
  parallel_num=1
fi

split -a 3 -d -n l/${parallel_num} eval_yuv_list.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./release/lib

rm -rf eval_result
mkdir eval_result
for k in $(seq 0 $((parallel_num - 1))); do
  {
    file=$(printf "x%03d" ${k})
    ./release/bin/example \
      --model_file=${runtime_model_file} \
      --model_name=${model_name} \
      --input_type=preprocessed_image \
      --input_config_string={\"image_list_file\":\"$file\"} \
      --output_type=raw \
      --output_config_string={\"output_file\":\"eval_result/${eval_result_file}.${k}\"}
  } &

done
function cleanup() {
  find . -type f -name "x[0-9]*" -exec rm -rf {} \;
  rm -rf eval_result
}

trap 'cleanup; kill 0' INT

wait

cat eval_result/* >${eval_result_file}
python3 -u ${eval_py} \
  --eval_result_path=${eval_result_file} \
  --annotation_path=${val_annotation_path}
cleanup
