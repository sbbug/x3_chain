#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.


cur_dir=$(
  cd "$(dirname $0)" || exit
  pwd
)

cd "${cur_dir}" || exit

source ../../env.conf
source ../env.conf

remote_host=$1

if [ -z $remote_host ]; then
  echo "Usage: sh 03_scp_to_board.sh \${board_ipaddr}"
  exit
fi

if [ ! -d release ]; then
  echo "Release not build."
  echo "please use [sh 01_build.sh] to build example."
  exit
fi

if [ ! -f ${cur_dir}/../mapper/model_output/${sample_name}_hybrid_horizonrt.bin ]; then
  echo "Runtime model not exist."
  echo "please use shell-scripts in mapper Directory to build runtime model."
  exit
fi

# clear workspace
ssh root@${remote_host} "rm -rf /userdata/samples/${sample_name}"
# create workspace
ssh root@${remote_host} "mkdir -p /userdata/samples/${sample_name}"

if [ -d ${cur_dir}/../config ]; then
    scp -r ${cur_dir}/../config root@${remote_host}:/userdata/samples/${sample_name}/
fi

scp ${cur_dir}/../mapper/model_output/${sample_name}_hybrid_horizonrt.bin root@${remote_host}:/userdata/samples/${sample_name}/
scp -r ${cur_dir}/release root@${remote_host}:/userdata/samples/${sample_name}/
scp ${cur_dir}/dev_board_*.sh root@${remote_host}:/userdata/samples/${sample_name}/
scp -r ${cur_dir}/../../../01_common/test_data root@${remote_host}:/userdata/samples/
cat ${cur_dir}/../../env.conf ${cur_dir}/../env.conf > tmp.conf
scp tmp.conf root@${remote_host}:/userdata/samples/${sample_name}/env.conf
rm -f tmp.conf
