// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/rfcn_post_process.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

int RfcnPostProcessModule::Init(std::string config_file,
                                std::string config_string) {
  DLOG(INFO) << "Init " << FullName();
  int ret_code = PostProcessModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

int RfcnPostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                       ImageTensor *image_tensor,
                                       Perception *perception) {
  perception->type = Perception::DET;
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  float *data = reinterpret_cast<float *>(tensor->data.virAddr);
  int *shape = tensor->data_shape.d;
  int len = shape[0] * shape[1] * shape[2] * shape[3];
  float scale_h =
      static_cast<float>(image_tensor->ori_height()) / image_tensor->height();
  float scale_w =
      static_cast<float>(image_tensor->ori_width()) / image_tensor->width();

  std::vector<Detection> &dets = perception->det;
  for (int i = 0; i < len; i = i + 6) {
    float score = data[i];
    if (score < 0.001) {
      break;  // end loop, end result array with score = 0
    }
    if (score < score_threshold_) {
      continue;
    }
    int id = static_cast<int>(data[i + 1]);
    if (id < 0 || id >= class_num_) {
      LOG(ERROR) << "Invalid id:" << id;
      continue;
    }
    float x1 = data[i + 2] * scale_w;
    float y1 = data[i + 3] * scale_h;
    float x2 = data[i + 4] * scale_w;
    float y2 = data[i + 5] * scale_h;
    Detection det;
    det.score = score;
    det.id = id;
    det.bbox.xmin = x1;
    det.bbox.ymin = y1;
    det.bbox.xmax = x2;
    det.bbox.ymax = y2;
    dets.push_back(det);
  }
  return 0;
}

int RfcnPostProcessModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (document.HasMember("class_num")) {
    class_num_ = document["class_num"].GetInt();
  }

  return 0;
}
