// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "glog/logging.h"
#include "input/feature_iterator.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

int FeatureIterator::Init(std::string config_file, std::string config_string) {
  int ret_code = DataIterator::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  ifs_.open(feature_list_file_, std::ios::in);
  if (!ifs_.is_open()) {
    LOG(ERROR) << "Open " << feature_list_file_ << " failed";
    return -1;
  }

  return 0;
}

bool FeatureIterator::Next(ImageTensor *image_tensor) {
  if (ifs_.eof()) {
    is_finish_ = true;
    return false;
  }

  std::string bin_file;
  ifs_ >> bin_file;

  if (bin_file.empty()) {
    return false;
  }

  // TODO(@horizon.ai):
  //  Here we use image tensor to represent feature data, it's not a good idea
  //
  std::vector<int> dims;
  ParsePathParams(bin_file, image_tensor->image_name, dims);
  auto &tensor = image_tensor->tensor;
  image_tensor->frame_id = NextFrameId();
  prepare_feature_tensor(dims, data_type_, &tensor);
  read_binary_file(bin_file, reinterpret_cast<char *>(tensor.data.virAddr));
  flush_tensor(&tensor);
  return true;
}
bool FeatureIterator::Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor){

     return true;
}
void FeatureIterator::Release(ImageTensor *image_tensor) {
  release_tensor(&image_tensor->tensor);
}

int FeatureIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("feature_list_file")) {
    feature_list_file_ = document["feature_list_file"].GetString();
  }

  if (document.HasMember("data_type")) {
    data_type_ =
        static_cast<hb_BPU_DATA_TYPE_E>(document["data_type"].GetInt());
  }

  return 0;
}

void FeatureIterator::ParsePathParams(std::string &input_file,
                                      std::string &image_name,
                                      std::vector<int> &dims) {
  //  {name}_{dim0}_{dim1}_{dim2}_{...}.bin
  std::string filename = get_file_name(input_file);

  std::vector<std::string> tokens;
  rsplit(filename, '.', tokens, 2);
  std::string field1 = tokens[1];

  tokens.resize(0);
  split(field1, '_', tokens);
  image_name = tokens[0];

  for (int i = 1; i < tokens.size(); i++) {
    dims.emplace_back(std::stoi(tokens[i]));
  }
}

FeatureIterator::~FeatureIterator() {
  if (ifs_.is_open()) {
    ifs_.close();
  }
}

bool FeatureIterator::HasNext() { return !is_finish_; }
