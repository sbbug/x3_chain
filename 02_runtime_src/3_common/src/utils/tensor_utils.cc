// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/tensor_utils.h"

#include <memory.h>

#include <iostream>

#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/image_utils.h"
#include "utils/utils.h"

void prepare_image_tensor(int height,
                          int width,
                          hb_BPU_DATA_TYPE_E data_type,
                          BPU_TENSOR_S *tensor) {
  BPU_DATA_TYPE_E image_data_type = data_type;
  tensor->data_type = image_data_type;
  int h_idx, w_idx, c_idx;
  auto ret =
      HB_BPU_getHWCIndex(tensor->data_type, nullptr, &h_idx, &w_idx, &c_idx);
  if (0 != ret) {
    LOG(FATAL) << "HB_BPU_getHWCIndex failed for image_data_type="
               << image_data_type << "!";
  }
  tensor->data_shape.ndim = 4;
  tensor->data_shape.d[0] = 1;
  tensor->data_shape.d[h_idx] = height;
  tensor->data_shape.d[w_idx] = width;
  tensor->data_shape.d[c_idx] = 3;
  tensor->aligned_shape = tensor->data_shape;
  if (image_data_type == BPU_TYPE_IMG_Y) {
    tensor->data_shape.d[c_idx] = 1;
    // Align by 16 bytes
    int stride = ALIGN_16(width);
    tensor->aligned_shape.d[w_idx] = stride;
    tensor->aligned_shape.d[c_idx] = 1;
    HB_SYS_bpuMemAlloc("in_data0", height * stride, true, &tensor->data);
  } else if (image_data_type == BPU_TYPE_IMG_YUV444 ||
             image_data_type == BPU_TYPE_IMG_BGR ||
             image_data_type == BPU_TYPE_IMG_RGB) {
    HB_SYS_bpuMemAlloc("in_data0", height * width * 3, true, &tensor->data);
  } else if (image_data_type == BPU_TYPE_IMG_YUV_NV12) {
    // Align by 16 bytes
    int stride = ALIGN_16(width);
    int y_length = height * stride;
    int uv_length = height / 2 * stride;
    tensor->aligned_shape.d[w_idx] = stride;
    HB_SYS_bpuMemAlloc("in_data0", y_length + uv_length, true, &tensor->data);
  } else if (image_data_type == BPU_TYPE_IMG_NV12_SEPARATE) {
    // Align by 16 bytes
    int stride = ALIGN_16(width);
    int y_length = height * stride;
    int uv_length = height / 2 * stride;
    tensor->aligned_shape.d[w_idx] = stride;
    HB_SYS_bpuMemAlloc("in_data0", y_length, true, &tensor->data);
    HB_SYS_bpuMemAlloc("in_data1", uv_length, true, &tensor->data_ext);
  } else if (image_data_type == BPU_TYPE_IMG_BGRP ||
             image_data_type == BPU_TYPE_IMG_RGBP) {
    int planar_length = height * width * 3;
    HB_SYS_bpuMemAlloc("in_data0", planar_length, true, &tensor->data);
  } else if (image_data_type == BPU_TYPE_TENSOR_F32 ||
             image_data_type == BPU_TYPE_TENSOR_S32 ||
             image_data_type == BPU_TYPE_TENSOR_U32) {
    HB_SYS_bpuMemAlloc("in_data0", height * width * 4, true, &tensor->data);
  } else if (image_data_type == BPU_TYPE_TENSOR_U8 ||
             image_data_type == BPU_TYPE_TENSOR_S8) {
    HB_SYS_bpuMemAlloc("in_data0", height * width, true, &tensor->data);
  } else {
    LOG(FATAL) << "Unimplemented for data type:" << image_data_type;
  }
}

void prepare_feature_tensor(std::vector<int> &dims,
                            hb_BPU_DATA_TYPE_E data_type,
                            BPU_TENSOR_S *tensor) {
  // make sure the layout is BPU_LAYOUT_NHWC
  tensor->data_shape.layout = BPU_LAYOUT_NHWC;
  tensor->data_type = data_type;
  tensor->data_shape.ndim = dims.size();
  int length = 1;
  for (int i = 0; i < dims.size(); i++) {
    tensor->data_shape.d[i] = dims[i];
    length *= dims[i];
  }
  tensor->aligned_shape = tensor->data_shape;
  switch (data_type) {
    case BPU_TYPE_TENSOR_U8:
    case BPU_TYPE_TENSOR_S8:
      length *= 1;
      break;
    case BPU_TYPE_TENSOR_F32:
    case BPU_TYPE_TENSOR_S32:
    case BPU_TYPE_TENSOR_U32:
      length *= 4;
      break;
    default:
      break;
  }
  HB_SYS_bpuMemAlloc("in_data0", length, true, &tensor->data);
}

int read_image_tensor(std::string &path,
                      int &ori_width,
                      int &ori_height,
                      BPU_TENSOR_S *tensor) {
  cv::Mat bgr_mat = cv::imread(path);
  ori_width = bgr_mat.cols;
  ori_height = bgr_mat.rows;

  auto data_type = tensor->data_type;
  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(tensor->data_type, nullptr, &h_idx, &w_idx, &c_idx);

  auto height = tensor->data_shape.d[h_idx];
  auto width = tensor->data_shape.d[w_idx];
  auto stride = tensor->aligned_shape.d[w_idx];

  cv::Mat resized_mat(height, width, bgr_mat.type());
  cv::resize(bgr_mat, resized_mat, resized_mat.size(), 0, 0);
  if (data_type == BPU_TYPE_IMG_Y) {
    cv::Mat gray;
    cv::cvtColor(resized_mat, gray, cv::COLOR_BGR2GRAY);
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor->data.virAddr);
    uint8_t *data = gray.data;
    for (int h = 0; h < height; ++h) {
      auto *raw = data0 + h * stride;
      for (int w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }
  } else if (data_type == BPU_TYPE_IMG_YUV_NV12) {
    cv::Mat nv12;
    bgr_to_nv12(resized_mat, nv12);
    uint8_t *data = nv12.data;

    // Copy y data to data0
    uint8_t *y = reinterpret_cast<uint8_t *>(tensor->data.virAddr);
    for (int h = 0; h < height; ++h) {
      auto *raw = y + h * stride;
      for (int w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }
    // Copy uv data to data1
    uint8_t *uv =
        reinterpret_cast<uint8_t *>(tensor->data.virAddr) + height * stride;
    memcpy(uv, nv12.data + height * width, height * width / 2);
  } else if (data_type == BPU_TYPE_IMG_NV12_SEPARATE) {
    cv::Mat nv12;
    bgr_to_nv12(resized_mat, nv12);
    uint8_t *data = nv12.data;

    // Copy y data to data0
    uint8_t *y = reinterpret_cast<uint8_t *>(tensor->data.virAddr);
    for (int h = 0; h < height; ++h) {
      auto *raw = y + h * stride;
      for (int w = 0; w < width; ++w) {
        *raw++ = *data++;
      }
    }
    // Copy uv data to data1
    uint8_t *uv = reinterpret_cast<uint8_t *>(tensor->data_ext.virAddr);
    memcpy(uv, nv12.data + height * width, height * width / 2);
  } else if (data_type == BPU_TYPE_IMG_YUV444) {
    cv::Mat yuv_mat;
    cv::cvtColor(resized_mat, yuv_mat, cv::COLOR_BGR2YUV);
    void *data = tensor->data.virAddr;
    memcpy(data, yuv_mat.ptr<uint8_t>(), height * width * 3);
  } else if (data_type == BPU_TYPE_IMG_BGR) {
    void *data = tensor->data.virAddr;
    memcpy(data, resized_mat.ptr<uint8_t>(), height * width * 3);
  } else if (data_type == BPU_TYPE_IMG_RGB) {
    cv::Mat rgb_mat;
    cv::cvtColor(resized_mat, rgb_mat, cv::COLOR_BGR2RGB);
    void *data = tensor->data.virAddr;
    memcpy(data, rgb_mat.ptr<uint8_t>(), height * width * 3);
  } else if (data_type == BPU_TYPE_IMG_BGRP) {
    int offset = height * width;
    nhwc_to_nchw(reinterpret_cast<uint8_t *>(tensor->data.virAddr),
                 reinterpret_cast<uint8_t *>(tensor->data.virAddr) + offset,
                 reinterpret_cast<uint8_t *>(tensor->data.virAddr) + offset * 2,
                 resized_mat.ptr<uint8_t>(),
                 height,
                 width);
  } else if (data_type == BPU_TYPE_IMG_RGBP) {
    cv::Mat rgb_mat;
    cv::cvtColor(resized_mat, rgb_mat, cv::COLOR_BGR2RGB);
    int offset = height * width;
    nhwc_to_nchw(reinterpret_cast<uint8_t *>(tensor->data.virAddr),
                 reinterpret_cast<uint8_t *>(tensor->data.virAddr) + offset,
                 reinterpret_cast<uint8_t *>(tensor->data.virAddr) + offset * 2,
                 rgb_mat.ptr<uint8_t>(),
                 height,
                 width);
  } else {
    LOG(ERROR) << "Un support model input data type: "
               << data_type_enum_to_string(data_type);
    return -1;
  }

  return 0;
}

void flush_tensor(BPU_TENSOR_S *tensor) {
  switch (tensor->data_type) {
    case BPU_TYPE_IMG_BGRP:
    case BPU_TYPE_IMG_RGBP:
    case BPU_TYPE_IMG_Y:
    case BPU_TYPE_IMG_YUV_NV12:
    case BPU_TYPE_IMG_RGB:
    case BPU_TYPE_IMG_BGR:
    case BPU_TYPE_IMG_YUV444:
    case BPU_TYPE_TENSOR_U8:
    case BPU_TYPE_TENSOR_S8:
    case BPU_TYPE_TENSOR_F32:
    case BPU_TYPE_TENSOR_S32:
    case BPU_TYPE_TENSOR_U32:
      HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_CLEAN);
      break;
    case BPU_TYPE_IMG_NV12_SEPARATE:
      HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_CLEAN);
      HB_SYS_flushMemCache(&(tensor->data_ext), HB_SYS_MEM_CACHE_CLEAN);
      break;
    default:
      break;
  }
}

void release_tensor(BPU_TENSOR_S *tensor) {
  switch (tensor->data_type) {
    case BPU_TYPE_IMG_BGRP:
    case BPU_TYPE_IMG_RGBP:
    case BPU_TYPE_IMG_Y:
    case BPU_TYPE_IMG_YUV_NV12:
    case BPU_TYPE_IMG_RGB:
    case BPU_TYPE_IMG_BGR:
    case BPU_TYPE_IMG_YUV444:
    case BPU_TYPE_TENSOR_U8:
    case BPU_TYPE_TENSOR_S8:
    case BPU_TYPE_TENSOR_F32:
    case BPU_TYPE_TENSOR_S32:
    case BPU_TYPE_TENSOR_U32:
      HB_SYS_bpuMemFree(&(tensor->data));
      break;
    case BPU_TYPE_IMG_NV12_SEPARATE:
      HB_SYS_bpuMemFree(&(tensor->data));
      HB_SYS_bpuMemFree(&(tensor->data_ext));
      break;
    default:
      break;
  }
}

void prepare_output_tensor(std::vector<BPU_TENSOR_S> &output,
                           BPU_MODEL_S *model) {
  int out_num = model->output_num;
  output.resize(out_num);
  std::string name_prefix("out_mem");
  for (int i = 0; i < out_num; ++i) {
    auto &out_node = (model->outputs)[i];
    int element_size = (out_node.data_type == BPU_TYPE_TENSOR_S8) ? 1 : 4;
    auto &aligned_shape = out_node.aligned_shape;
    int out_aligned_size = element_size;
    for (int j = 0; j < aligned_shape.ndim; ++j) {
      out_aligned_size = out_aligned_size * aligned_shape.d[j];
    }
    std::string mem_name = name_prefix + std::to_string(i);
    output[i].data_shape = out_node.shape;
    output[i].aligned_shape = out_node.aligned_shape;
    output[i].data_type = out_node.data_type;
    // TODO(@horizon.ai): shifts data for tensor (only need by quanti model)
    auto &tensor_data = output[i].data;
    HB_SYS_bpuMemAlloc(mem_name.data(), out_aligned_size, true, &tensor_data);
  }
}

void release_output_tensor(std::vector<BPU_TENSOR_S> &output) {
  for (auto tensor : output) {
    HB_SYS_bpuMemFree(&(tensor.data));
  }
}
