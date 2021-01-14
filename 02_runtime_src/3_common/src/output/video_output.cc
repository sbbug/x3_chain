// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "output/video_output.h"

#include <fstream>

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "utils/image_utils.h"

int VideoOutputModule::Init(std::string config_file,
                            std::string config_string) {
  int ret_code = OutputModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  return 0;
}

void VideoOutputModule::Write(ImageTensor *frame, Perception *perception) {
  cv::Mat mat;
  if (draw_perception(frame, perception, mat) == 0) {
    video_writer_.write(mat);
  }
}

int VideoOutputModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("fps")) {
    fps_ = document["fps"].GetInt();
  }

  if (document.HasMember("video_name")) {
    video_name_ = document["video_name"].GetString();
  }

  if (document.HasMember("height")) {
    height_ = document["height"].GetInt();
  }

  if (document.HasMember("width")) {
    width_ = document["width"].GetInt();
  }

  video_writer_.open(video_name_,
                     CV_FOURCC('M', 'J', 'P', 'G'),
                     fps_,
                     cv::Size(width_, height_));
  if (!video_writer_.isOpened()) {
    return -1;
  }

  return 0;
}

VideoOutputModule::~VideoOutputModule() {
  if (video_writer_.isOpened()) {
    video_writer_.release();
  }
}
