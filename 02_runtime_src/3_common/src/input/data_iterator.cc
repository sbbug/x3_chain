// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/data_iterator.h"

#include "glog/logging.h"
#include "input/camera_data_iterator.h"
#include "input/feature_iterator.h"
#include "input/image_list_data_iterator.h"
#include "input/network_data_iterator.h"
#include "input/preprocessed_image_iterator.h"
#include "input/mutil_modal_image_list_data_iterator.h"

int DataIterator::Init(std::string config_file, std::string config_string) {
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

int DataIterator::LoadConfigFile(std::string& config_file) {
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

DataIterator* DataIterator::GetImpl(const std::string& module_name) {
  if (module_name == "image") {
    return new ImageListDataIterator();
  } else if (module_name == "mutil_modal") {
    return new MutilModalImageListDataIterator();
  }else if (module_name == "network") {
    return new NetworkDataIterator();
  } else if (module_name == "preprocessed_image") {
    return new PreprocessedImageIterator();
  } else if (module_name == "camera") {
    return new CameraDataIterator();
  } else if (module_name == "feature") {
    return new FeatureIterator;
  } else {
    LOG(FATAL) << "Unsupported module:" << module_name;
  }
}
