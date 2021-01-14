#!/usr/bin/env sh
# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

 
pid=$(ps | grep release | grep -v grep | awk '{print $1}')
echo PID_is_$pid
if [ -z "$pid" ]
then
      echo "pid is empty, process already killed"
else
      kill -9 $pid
fi
