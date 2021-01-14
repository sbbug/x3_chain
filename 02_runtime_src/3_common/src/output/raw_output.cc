// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "output/raw_output.h"

#include <iterator>

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

int RawOutputModule::Init(std::string config_file, std::string config_string) {
  int ret_code = OutputModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  ofs_.open(output_file_.c_str(), std::ios::out | std::ios::trunc);
  if (!ofs_.is_open()) {
    LOG(ERROR) << "Open " << config_file << " failed";
  }
  return 0;
}

void RawOutputModule::Write(ImageTensor *frame, Perception *perception) {
  ofs_ << "{"
       << R"("frame")"
       << ":" << *frame << ","
       << R"("result")"
       << ":" << *perception << "}" << std::endl;
}

int RawOutputModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("output_file")) {
    output_file_ = document["output_file"].GetString();
  }

  return 0;
}

RawOutputModule::~RawOutputModule() {
  if (ofs_.is_open()) {
    ofs_.close();
  }
}
