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


eval_result_file=${sample_name}_eval.out
#scp root@$1:/userdata/samples/${sample_name}/${eval_result_file} ./${eval_result_file}

python3 -u ${eval_py} \
  --eval_result_path=${eval_result_file} \
  --annotation_path=${val_annotation_path}
