// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/image_list_data_iterator.h"

#include <iostream>

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

int ImageListDataIterator::Init(std::string config_file,
                                std::string config_string) {
  DLOG(INFO) << "Init image list data iterator";
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

bool ImageListDataIterator::Next(ImageTensor *image_tensor) {
  if (ifs_.eof()) {
    is_finish_ = true;
    return false;
  }

  std::string image_file;
  ifs_ >> image_file;

  if (image_file.empty()) {
    return false;
  }

  auto &tensor = image_tensor->tensor;
  prepare_image_tensor(height_, width_, data_type_, &tensor);
  read_image_tensor(image_file,
                    image_tensor->ori_image_width,
                    image_tensor->ori_image_height,
                    &tensor);
  image_tensor->image_name = get_file_name(image_file);
  image_tensor->timestamp = Stopwatch::CurrentTs();
  image_tensor->frame_id = NextFrameId();
  flush_tensor(&tensor);
  return true;
}
bool ImageListDataIterator::Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor){
if (ifs_.eof()) {
    is_finish_ = true;
    return false;
  }

  std::string image_file;
  ifs_ >> image_file;

  std::vector<std::string> pairs = s_split(image_file,";");
  std::string visible_image_file = pairs[0];
  std::string lwir_image_file = pairs[1];

  std::cout<<visible_image_file<<std::endl;
  std::cout<<lwir_image_file<<std::endl;
  if (visible_image_file.empty() || lwir_image_file.empty()) {
    return false;
  }

  auto &visible_tensor = visible_image_tensor->tensor;
  auto &lwir_tensor = lwir_image_tensor->tensor;
  prepare_image_tensor(height_, width_, data_type_, &visible_tensor);
  prepare_image_tensor(height_, width_, data_type_, &lwir_tensor);

  read_image_tensor(visible_image_file,
                    visible_image_tensor->ori_image_width,
                    visible_image_tensor->ori_image_height,
                    &visible_tensor);
  read_image_tensor(lwir_image_file,
                    lwir_image_tensor->ori_image_width,
                    lwir_image_tensor->ori_image_height,
                    &lwir_tensor);

  visible_image_tensor->image_name = get_file_name(visible_image_file);
  lwir_image_tensor->image_name = get_file_name(lwir_image_file);
  visible_image_tensor->timestamp = Stopwatch::CurrentTs();
  lwir_image_tensor->timestamp = visible_image_tensor->timestamp;
  visible_image_tensor->frame_id = NextFrameId();
  lwir_image_tensor->frame_id = visible_image_tensor->frame_id;
  flush_tensor(&visible_tensor);
  flush_tensor(&lwir_tensor);
  return true;
}
void ImageListDataIterator::Release(ImageTensor *image_tensor) {
  release_tensor(&image_tensor->tensor);
}

int ImageListDataIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("image_list_file")) {
    image_list_file_ = document["image_list_file"].GetString();
  }

  if (document.HasMember("width")) {
    width_ = document["width"].GetInt();
  }

  if (document.HasMember("height")) {
    height_ = document["height"].GetInt();
  }

  if (document.HasMember("data_type")) {
    data_type_ =
        static_cast<hb_BPU_DATA_TYPE_E>(document["data_type"].GetInt());
  }

  return 0;
}

ImageListDataIterator::~ImageListDataIterator() {
  if (ifs_.is_open()) {
    ifs_.close();
  }
}

bool ImageListDataIterator::HasNext() { return !is_finish_; }
