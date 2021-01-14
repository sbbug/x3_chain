// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/preprocessed_image_iterator.h"

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

int PreprocessedImageIterator::Init(std::string config_file,
                                    std::string config_string) {
  int ret_code = DataIterator::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  ifs_.open(image_list_file_, std::ios::in);
  if (!ifs_.is_open()) {
    LOG(ERROR) << "Open " << image_list_file_ << " failed";
    return -1;
  }

  return 0;
}

bool PreprocessedImageIterator::Next(ImageTensor *image_tensor) {
  if (ifs_.eof()) {
    is_finish_ = true;
    return false;
  }

  std::string bin_file;
  ifs_ >> bin_file;

  if (bin_file.empty()) {
    return false;
  }

  int input_height;
  int input_width;
  ParsePathParams(bin_file,
                  image_tensor->image_name,
                  image_tensor->ori_image_height,
                  image_tensor->ori_image_width,
                  input_height,
                  input_width);

  // TODO(@horizon.ai): checking this
  image_tensor->is_pad_resize = true;
  auto &tensor = image_tensor->tensor;
  image_tensor->frame_id = NextFrameId();
  prepare_image_tensor(input_height, input_width, data_type_, &tensor);
  read_binary_file(bin_file, reinterpret_cast<char *>(tensor.data.virAddr));
  flush_tensor(&tensor);
  return true;
}
bool PreprocessedImageIterator::Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor){

     return true;
}
void PreprocessedImageIterator::Release(ImageTensor *image_tensor) {
  release_tensor(&image_tensor->tensor);
}

int PreprocessedImageIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("image_list_file")) {
    image_list_file_ = document["image_list_file"].GetString();
  }

  if (document.HasMember("data_type")) {
    data_type_ =
        static_cast<hb_BPU_DATA_TYPE_E>(document["data_type"].GetInt());
  }

  return 0;
}

void PreprocessedImageIterator::ParsePathParams(std::string &input_file,
                                                std::string &image_name,
                                                int &org_h,
                                                int &org_w,
                                                int &dst_h,
                                                int &dst_w) {
  // {name}_{org_h}_{org_w}_{dst_h}_{dst_w}.bin
  std::string filename = get_file_name(input_file);

  std::vector<std::string> tokens;
  rsplit(filename, '.', tokens, 2);
  std::string field1 = tokens[1];

  tokens.resize(0);
  rsplit(field1, '_', tokens, 5);

  dst_w = std::stoi(tokens[0]);
  dst_h = std::stoi(tokens[1]);
  org_w = std::stoi(tokens[2]);
  org_h = std::stoi(tokens[3]);
  image_name = tokens[4];
}

PreprocessedImageIterator::~PreprocessedImageIterator() {
  if (ifs_.is_open()) {
    ifs_.close();
  }
}

bool PreprocessedImageIterator::HasNext() { return !is_finish_; }
