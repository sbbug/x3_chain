// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_INPUT_DATA_H_
#define _INPUT_INPUT_DATA_H_

#include <ostream>
#include <string>

#include "bpu_io.h"
#include "bpu_predict_extension.h"
#include "glog/logging.h"

typedef struct ImageTensor {
  BPU_TENSOR_S tensor;
  int32_t frame_id = 0;
  int32_t cam_id = 0;
  uint64_t timestamp = 0;
  std::string image_name;
  int ori_image_width;
  int ori_image_height;

  void *addition_info;

  // TODO(@horizon.ai):
  bool is_pad_resize = false;

  inline int32_t height() {
    int height, width;
    if (0 !=
        HB_BPU_getHW(tensor.data_type, &tensor.data_shape, &height, &width)) {
      LOG(FATAL) << "HB_BPU_getHW failed: data_type:" << tensor.data_type
                 << ", data_shape:layout:" << tensor.data_shape.layout
                 << ",ndim:" << tensor.data_shape.ndim
                 << "dims:" << tensor.data_shape.d;
    }
    return height;
  }

  inline int32_t width() {
    int height, width;
    if (0 !=
        HB_BPU_getHW(tensor.data_type, &tensor.data_shape, &height, &width)) {
      LOG(FATAL) << "HB_BPU_getHW failed: data_type:" << tensor.data_type
                 << ", data_shape:layout:" << tensor.data_shape.layout
                 << ",ndim:" << tensor.data_shape.ndim
                 << "dims:" << tensor.data_shape.d;
    }
    return width;
  }

  inline int32_t ori_height() { return ori_image_height; }

  inline int32_t ori_width() { return ori_image_width; }

  friend std::ostream &operator<<(std::ostream &os, ImageTensor &image_tensor) {
    os << "{"
       << R"("image_name")"
       << ":\"" << image_tensor.image_name << "\", "
       << R"("image_width")"
       << ":" << image_tensor.ori_image_width << ", "
       << R"("image_height")"
       << ":" << image_tensor.ori_image_height << "}";
    return os;
  }
} ImageTensor;

#endif  // _INPUT_INPUT_DATA_H_
