// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/utils.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <regex>
#include "glog/logging.h"
#include "utils/tensor_utils.h"

int read_binary_file(std::string &file_path, char **bin, int *length) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    LOG(ERROR) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  *length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  *bin = new char[sizeof(char) * (*length)];
  ifs.read(*bin, *length);
  ifs.close();
  return 0;
}

int read_binary_file(std::string &file_path, char *bin) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    LOG(ERROR) << "Open " << file_path << " failed";
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  ifs.read(bin, length);
  ifs.close();
  return 0;
}

int read_binary_file(std::string &file_path,
                     BPU_TENSOR_S *input_tensor,
                     char *bin) {
  BPU_DATA_TYPE_E image_data_type = input_tensor->data_type;
  int h_idx, w_idx, c_idx;
  auto ret = HB_BPU_getHWCIndex(
      input_tensor->data_type, nullptr, &h_idx, &w_idx, &c_idx);
  int height = input_tensor->data_shape.d[h_idx];
  int width = input_tensor->data_shape.d[w_idx];
  if (image_data_type == BPU_TYPE_IMG_Y) {
    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      LOG(ERROR) << "Open " << file_path << " failed";
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    std::cout << "file length: " << length << std::endl;
    ifs.seekg(0, std::ios::beg);
    char *tmp_char = new char[height * width];
    if (height * width != length) {
      std::cout << "file length does not match model input \n";
    }
    ifs.read(tmp_char, height * width);
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        bin[h * ALIGN_16(width) + w] = tmp_char[h * width + w];
      }
    }
    ifs.close();
    delete[] tmp_char;
    return 0;
  } else if (image_data_type == BPU_TYPE_IMG_YUV444) {
    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      LOG(ERROR) << "Open " << file_path << " failed";
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    ifs.read(bin, length);
    ifs.close();
    return 0;
  } else {
    std::cout << "wrong img type, this function only support gray and yuv444 "
                 "for now\n";
    return -1;
  }
}

int load_model_from_file(std::string &model_file, BPU_MODEL_S *bpu_model) {
  // Read model file
  char *model_bin;
  int model_length;
  int ret_code = read_binary_file(model_file, &model_bin, &model_length);
  if (ret_code != 0) {
    LOG(ERROR) << "Read " << model_file << " failed";
    return ret_code;
  }

  // Load model
  ret_code = HB_BPU_loadModel(model_bin, model_length, bpu_model);
  LOG_IF(ERROR, ret_code != 0)
      << "Load model failed, " << HB_BPU_getErrorName(ret_code);
  delete[] model_bin;
  return ret_code;
}

std::string data_type_enum_to_string(BPU_DATA_TYPE_E data_type) {
  switch (data_type) {
    case BPU_TYPE_IMG_Y:
      return "BPU_TYPE_IMG_Y";
    case BPU_TYPE_IMG_YUV_NV12:
      return "BPU_TYPE_IMG_YUV_NV12";
    case BPU_TYPE_IMG_YUV444:
      return "BPU_TYPE_IMG_YUV444";
    case BPU_TYPE_IMG_BGR:
      return "BPU_TYPE_IMG_BGR";
    case BPU_TYPE_IMG_BGRP:
      return "BPU_TYPE_IMG_BGRP";
    case BPU_TYPE_IMG_RGB:
      return "BPU_TYPE_IMG_RGB";
    case BPU_TYPE_IMG_RGBP:
      return "BPU_TYPE_IMG_RGBP";
    case BPU_TYPE_TENSOR_U8:
      return "BPU_TYPE_TENSOR_U8";
    case BPU_TYPE_TENSOR_S8:
      return "BPU_TYPE_TENSOR_S8";
    case BPU_TYPE_TENSOR_F32:
      return "BPU_TYPE_TENSOR_F32";
    case BPU_TYPE_TENSOR_S32:
      return "BPU_TYPE_TENSOR_S32";
    case BPU_TYPE_TENSOR_U32:
      return "BPU_TYPE_TENSOR_U32";
    case BPU_TYPE_MAX:
      return "BPU_TYPE_MAX";
  }
}

std::string layout_type_enum_to_string(BPU_LAYOUT_E layout) {
  switch (layout) {
    case BPU_LAYOUT_NCHW:
      return "BPU_LAYOUT_NCHW";
    case BPU_LAYOUT_NHWC:
      return "BPU_LAYOUT_NHWC";
  }
}

std::string get_file_name(std::string &path) {
  int slash_pos = path.rfind('/');
  return path.substr(slash_pos + 1);
}

void split(std::string &str,
           char sep,
           std::vector<std::string> &tokens,
           int limit) {
  int pos = -1;
  while (true) {
    int next_pos = str.find(sep, pos + 1);
    if (next_pos == std::string::npos) {
      tokens.emplace_back(str.substr(pos + 1));
      break;
    }
    tokens.emplace_back(str.substr(pos + 1, next_pos - pos - 1));
    if (tokens.size() == limit - 1) {
      tokens.emplace_back(str.substr(next_pos + 1));
      break;
    }
    pos = next_pos;
  }
}

void rsplit(std::string &str,
            char sep,
            std::vector<std::string> &tokens,
            int limit) {
  int pos = str.size();
  while (true) {
    int prev_pos = str.rfind(sep, pos - 1);
    if (prev_pos == std::string::npos) {
      tokens.emplace_back(str.substr(0, pos));
      break;
    }

    tokens.emplace_back(str.substr(prev_pos + 1, pos - prev_pos - 1));
    if (tokens.size() == limit - 1) {
      tokens.emplace_back(str.substr(0, prev_pos));
      break;
    }

    pos = prev_pos;
  }
}

void nhwc_to_nchw(uint8_t *out_data0,
                  uint8_t *out_data1,
                  uint8_t *out_data2,
                  uint8_t *in_data,
                  int height,
                  int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data0++ = *(in_data++);
      *out_data1++ = *(in_data++);
      *out_data2++ = *(in_data++);
    }
  }
}

void nchw_to_nhwc(uint8_t *out_data,
                  uint8_t *in_data0,
                  uint8_t *in_data1,
                  uint8_t *in_data2,
                  int height,
                  int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data++ = *(in_data0++);
      *out_data++ = *(in_data1++);
      *out_data++ = *(in_data2++);
    }
  }
}

bool operator==(BPU_DATA_SHAPE_S &lhs, BPU_DATA_SHAPE_S &rhs) {
  if (lhs.ndim != rhs.ndim) return false;
  for (int i = 0; i < lhs.ndim; i++) {
    if (lhs.d[i] != rhs.d[i]) {
      return false;
    }
  }
  return true;
}

BPU_DATA_SHAPE_S squeeze(BPU_DATA_SHAPE_S &shape) {
  BPU_DATA_SHAPE_S s;
  s.layout = shape.layout;
  s.ndim = 0;
  for (int i = 0; i < shape.ndim; i++) {
    if (shape.d[i] != 1) {
      s.d[s.ndim++] = shape.d[i];
    }
  }
  return s;
}

std::string model_info(BPU_MODEL_S *bpu_model) {
  auto shape_str_fn = [](BPU_DATA_SHAPE_S *shape) {
    std::stringstream ss;
    ss << "(";
    std::copy(
        shape->d, shape->d + shape->ndim, std::ostream_iterator<int>(ss, ","));
    ss << ")";
    ss << ", layout:" << layout_type_enum_to_string(shape->layout);
    return ss.str();
  };

  std::stringstream ss;
  ss << "Input num:" << bpu_model->input_num;
  for (int i = 0; i < bpu_model->input_num; i++) {
    auto &input_node = bpu_model->inputs[i];
    ss << ", input[" << i << "]: "
       << "name:" << input_node.name
       << ", data type:" << data_type_enum_to_string(input_node.data_type)
       << ", shape:" << shape_str_fn(&input_node.shape)
       << ", aligned shape:" << shape_str_fn(&input_node.aligned_shape);
  }

  ss << ", Output num:" << bpu_model->output_num;
  for (int i = 0; i < bpu_model->output_num; i++) {
    auto &output_node = bpu_model->outputs[i];
    ss << ", output[" << i << "]: "
       << "name:" << output_node.name << ", op:" << output_node.op_type
       << ", data type:" << data_type_enum_to_string(output_node.data_type)
       << ", shape:" << shape_str_fn(&output_node.shape)
       << ", aligned shape:" << shape_str_fn(&output_node.aligned_shape);
  }

  return ss.str();
}
std::vector<std::string> s_split(const std::string& in, const std::string& delim) {
    std::regex re{ delim };
    return std::vector<std::string> {
        std::sregex_token_iterator(in.begin(), in.end(), re, -1),
            std::sregex_token_iterator()
    };
}