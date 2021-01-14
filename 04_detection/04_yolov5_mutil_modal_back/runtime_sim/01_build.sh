#!/usr/bin/env bash
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

cur_path=$(
  cd "$(dirname $0)" || exit
  pwd
)
cd ${cur_path} || exit

source ../../env.conf
source ../env.conf

if [ ! -d release ]; then
  mkdir release
fi

cd ../../../02_runtime_src || exit
set +e
lsb_check=$(which lsb_release)
if [ "$?" = '0' ]
then
  sys_id=$(lsb_release -a | grep "Distributor ID" | awk -F ':' '{print $2}')
  sys_id="$(echo -e "${sys_id}" | tr -d '[:space:]')"
else
  echo "can not find system type, assume system is CentOS as default"
  sys_id=default
fi
echo system is : $sys_id
  
if  [ "$sys_id" == "Ubuntu" ]; then
    echo "System is Ubuntu"
    if [ -d "deps/x86/gflags_ubuntu" ]; then
        rm -rf deps/x86/gflags
        mv deps/x86/gflags_ubuntu deps/x86/gflags
    fi
    if [ -d "deps/x86/protobuf_ubuntu" ]; then
        rm -rf deps/x86/protobuf
        mv deps/x86/protobuf_ubuntu deps/x86/protobuf
    fi
fi
set -e
rm -rf build_sim_${sample_name}
mkdir build_sim_${sample_name}
cd build_sim_${sample_name} || exit
cmake .. -DPLATFORM=x86 -DSDK_EXAMPLE=OFF -DCMAKE_INSTALL_PREFIX="${cur_path}/release"

make -j$(nproc)
make install
