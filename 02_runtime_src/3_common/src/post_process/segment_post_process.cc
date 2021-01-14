// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/segment_post_process.h"

#include <base/perception_common.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

int SegmentPostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                          ImageTensor *image_tensor,
                                          Perception *perception) {
  perception->type = Perception::SEG;
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  // int *shape = tensor->data_shape.d;
  int height, width;
  HB_BPU_getHW(tensor->data_type, &tensor->data_shape, &height, &width);
  // int input_height = shape[1];
  // int input_width = shape[2];
  int ori_image_height = image_tensor->ori_height();
  int ori_image_width = image_tensor->ori_width();
  float *data = reinterpret_cast<float *>(tensor->data.virAddr);
  cv::Mat result;
  result.create(height, width, CV_8UC1);
  for (int h = 0; h < height; ++h) {
    uint8_t *ptr = result.ptr<uint8_t>(h);
    for (int w = 0; w < width; ++w) {
      uint8_t class_id = static_cast<uint8_t>(data[h * width + w]);
      *(ptr + w) = class_id;
    }
  }
  cv::resize(result, result, cv::Size(ori_image_width, ori_image_height));
  perception->seg.resize(ori_image_width * ori_image_height);
  memcpy(
      perception->seg.data(), result.data, ori_image_width * ori_image_height);
  return 0;
}
