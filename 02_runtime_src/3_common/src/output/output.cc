// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "output/output.h"

#include "glog/logging.h"
#include "output/client_output.h"
#include "output/image_list_output.h"
#include "output/raw_output.h"
#include "output/video_output.h"

int OutputModule::Init(std::string config_file, std::string config_string) {
  if (!config_file.empty()) {
    int ret_code = this->LoadConfigFile(config_file);
    if (ret_code != 0) {
      return ret_code;
    }
  }

  if (!config_string.empty()) {
    int ret_code = this->LoadConfig(config_string);
    if (ret_code != 0) {
      return ret_code;
    }
  }

  return 0;
}

int OutputModule::LoadConfigFile(std::string& config_file) {
  std::ifstream ifs(config_file.c_str());
  if (!ifs) {
    LOG(ERROR) << "Open config file " << config_file << " failed";
    return -1;
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string contents(buffer.str());
  return this->LoadConfig(contents);
}

OutputModule* OutputModule::GetImpl(const std::string& module_name) {
  if (module_name == "raw") {
    return new RawOutputModule;
  } else if (module_name == "image") {
    return new ImageListOutputModule;
  } else if (module_name == "video") {
    return new VideoOutputModule;
  } else if (module_name == "client") {
    return new ClientOutputModule;
  }
  return NULL;
}
