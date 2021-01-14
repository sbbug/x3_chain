// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/post_process.h"

#include "glog/logging.h"
#include "post_process/classification_post_process.h"
#include "post_process/fasterrcnn_post_process.h"
#include "post_process/rfcn_post_process.h"
#include "post_process/s3fd_post_process.h"
#include "post_process/segment_post_process.h"
#include "post_process/ssd_post_process.h"
#include "post_process/yolo2_post_process.h"
#include "post_process/yolo3_post_process.h"
#include "post_process/yolo5_post_process.h"

int PostProcessModule::Init(std::string config_file,
                            std::string config_string) {
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

int PostProcessModule::LoadConfigFile(std::string &config_file) {
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

std::string PostProcessModule::FullName() {
  return module_name_ + ":" + instance_name_;
}

PostProcessModule *PostProcessModule::GetImpl(const std::string &model_name) {
  if (model_name == "fasterrcnn") {
    return new FasterrcnnPostProcessModule(model_name);
  } else if (model_name == "s3fd") {
    return new S3fdPostProcessModule(model_name);
  } else if (model_name == "ssd") {
    return new SsdPostProcessModule(model_name);
  } else if (model_name == "yolov2") {
    return new Yolo2PostProcessModule(model_name);
  } else if (model_name == "yolov3") {
    return new Yolo3PostProcessModule(model_name);
  } else if (model_name == "yolov5") {
    return new Yolo5PostProcessModule(model_name);
  } else if (model_name == "rfcn") {
    return new RfcnPostProcessModule(model_name);
  } else if (model_name == "segment") {
    return new SegmentPostProcessModule(model_name);
  } else if (model_name == "lenet" || model_name == "resnet18" || model_name == "googlenet" ||
             model_name == "se_resnet_gray" || model_name == "mobilenetv1" ||
             model_name == "mobilenetv2" || model_name == "resnet50_feature" ||
             model_name == "lenet_gray" || model_name == "classification" ||
             model_name.find("efficientnet") != std::string::npos) {
    return new ClassificationPostProcessModule(model_name);
  } else {
    LOG(ERROR) << "unspport model type:" << model_name;
    return nullptr;
  }
}
