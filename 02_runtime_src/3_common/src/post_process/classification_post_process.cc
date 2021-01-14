// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/classification_post_process.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "glog/logging.h"
#include "rapidjson/document.h"

int ClassificationPostProcessModule::Init(std::string config_file,
                                          std::string config_string) {
  DLOG(INFO) << "Init " << FullName();
  int ret_code = PostProcessModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

int ClassificationPostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                                 ImageTensor *image_tensor,
                                                 Perception *perception) {
  perception->type = Perception::CLS;
  if (top_k_ == 1) {
    perception->cls.resize(1);
    GetMaxResult(tensor, &perception->cls[0]);
  } else {
    GetTopkResult(tensor, perception->cls);
  }
  return 0;
}

void ClassificationPostProcessModule::GetMaxResult(BPU_TENSOR_S *tensor,
                                                   Classification *cls) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  float *scores = reinterpret_cast<float *>(tensor->data.virAddr);
  //float score = 0;
  float score = -1061109568;
  int id = 0;
  int *shape = tensor->data_shape.d;
  for (auto i = 0; i < shape[1] * shape[2] * shape[3]; i++) {
    if (scores[i] > score) {
      score = scores[i];
      id = i;
    }
  }
  cls->id = id;
  cls->score = score;
  cls->class_name = GetClsName(id);
}

void ClassificationPostProcessModule::GetTopkResult(
    BPU_TENSOR_S *tensor, std::vector<Classification> &top_k_cls) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  float *scores = reinterpret_cast<float *>(tensor->data.virAddr);
  auto cmp = std::greater<Classification>();

  // Get top k scores by min heap
  int *shape = tensor->data_shape.d;
  for (auto i = 0; i < shape[1] * shape[2] * shape[3]; i++) {
    float score = scores[i];
    if (top_k_cls.size() < top_k_) {
      // Heap is not full, just push heap
      top_k_cls.emplace_back(Classification(i, score, GetClsName(i)));
      std::push_heap(top_k_cls.begin(), top_k_cls.end(), cmp);
    } else {
      // Heap is full, pop heap & push heap if score greater than heap top
      // We reuse last element here
      std::pop_heap(top_k_cls.begin(), top_k_cls.end(), cmp);
      auto &last = top_k_cls.back();
      if (last.score < score) {
        last.id = i;
        last.score = score;
      }
      std::push_heap(top_k_cls.begin(), top_k_cls.end(), cmp);
    }
  }
  // Finally, sort heap
  std::sort(top_k_cls.begin(), top_k_cls.end(), cmp);
}

const char *ClassificationPostProcessModule::GetClsName(int id) {
  if (!class_names_.empty()) {
    return class_names_[id].c_str();
  }
  return nullptr;
}

int ClassificationPostProcessModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }
  if (document.HasMember("top_k")) {
    top_k_ = document["top_k"].GetInt();
  }

  if (document.HasMember("class_names")) {
    auto class_arr = document["class_names"].GetArray();
    class_names_.resize(class_arr.Size());
    for (int i = 0; i < class_arr.Size(); i++) {
      class_names_[i] = class_arr[i].GetString();
    }
  }
  return 0;
}
