#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

export HORIZON_RUN_CONTEXT=ARM
cd "$(dirname $0)" || exit

source ../../env.conf
source ../env.conf

set -e
if [ -z "$1" ]; then
  echo "usage: sh $0 {ip}"
  exit
fi

ip=$1

image_count=100
if [ "$TEST_CTX" == "SMOKE" ]; then
    image_count=10
fi

trap "ssh root@${ip} 'sh /userdata/samples/${sample_name}/dev_board_kill.sh'" 0

perf_result_file=${sample_name}_perf.txt
ssh root@${ip} "sh /userdata/samples/${sample_name}/dev_board_03_perf.sh " > ${perf_result_file} 2>&1

echo "===REPORT-START{RUNTIME-ARM-PERF}==="
cat ${perf_result_file} | grep "Whole process statistics" | awk -F] '{print $2}'
cat ${perf_result_file} | grep -A 1 "Infer stage statistics"
echo "===REPORT-END{RUNTIME-ARM-PERF}==="
