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

cd "${cur_path}" || exit

source ../../env.conf
source ../env.conf

if [ ! -d release ]; then
  mkdir release
fi

cd ../../../02_runtime_src || exit
rm -rf build_arm_${sample_name}
mkdir build_arm_${sample_name}

cd build_arm_${sample_name} || exit
cmake .. -DPLATFORM=arm -DSDK_EXAMPLE=OFF -DCMAKE_INSTALL_PREFIX="${cur_path}/release"

make -j$(nproc)
make install
