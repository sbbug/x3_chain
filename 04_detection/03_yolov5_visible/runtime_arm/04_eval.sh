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

set -ex
if [ -z "$1" ]; then
  echo "usage: sh $0 {ip}"
  exit
fi

image_count=50000
if [ "$TEST_CTX" == "SMOKE" ]; then
    image_count=10
fi

ip=$1

trap "ssh root@${ip} 'sh /userdata/samples/${sample_name}/dev_board_kill.sh'" 0

ssh root@${ip} "nohup sh /userdata/samples/${sample_name}/dev_board_02_eval.sh > /dev/null 2>&1 &"
bash send_image.sh ${ip} ${image_count}
sleep 1
bash calc_eval_acc.sh ${ip}
