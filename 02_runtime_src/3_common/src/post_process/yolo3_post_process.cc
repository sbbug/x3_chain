// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/yolo3_post_process.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "base/perception_common.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "utils/algorithm.h"
#include "utils/nms.h"

Yolo3Config default_yolo3_config = {
    {32, 16, 8},
    {{{3.625, 2.8125}, {4.875, 6.1875}, {11.65625, 10.1875}},
     {{1.875, 3.8125}, {3.875, 2.8125}, {3.6875, 7.4375}},
     {{1.25, 1.625}, {2.0, 3.75}, {4.125, 2.875}}},
    10,
    { "pedestrian",      "people",
     "bicycle",    "car",     "van",
     "truck",         "tricycle",        "awning-tricycle",
     "bus", "motor", 

   }};

//Yolo3Config default_yolo3_config = {
//    {32, 16, 8},
//    {{{3.625, 2.8125}, {4.875, 6.1875}, {11.65625, 10.1875}},
//     {{1.875, 3.8125}, {3.875, 2.8125}, {3.6875, 7.4375}},
//     {{1.25, 1.625}, {2.0, 3.75}, {4.125, 2.875}}},
//    20,
//    {"Expressway-Service-area",        "Expressway-toll-station",      "airplane",
//     "airport",    "baseballfield",     "basketballcourt","bridge","chimney","dam",
//     "golffield",         "groundtrackfield",        "harbor","overpass","ship","stadium",
//     "storagetank", "tenniscourt", "trainstation","vehicle","windmill"
//
//   }};
int Yolo3PostProcessModule::Init(std::string config_file,
                                 std::string config_string) {
  DLOG(INFO) << "Init " << FullName();
  int ret_code = PostProcessModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

void Yolo3PostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                         ImageTensor *frame,
                                         int layer,
                                         std::vector<Detection> &dets) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<float *>(tensor->data.virAddr);
  int num_classes = yolo3_config_.class_num;
  int stride = yolo3_config_.strides[layer];
  int num_pred = yolo3_config_.class_num + 4 + 1;

  std::vector<float> class_pred(yolo3_config_.class_num, 0.0);
  std::vector<std::pair<double, double>> &anchors =
      yolo3_config_.anchors_table[layer];

  double h_ratio = frame->height() * 1.0 / frame->ori_height();
  double w_ratio = frame->width() * 1.0 / frame->ori_width();
  double resize_ratio = std::min(w_ratio, h_ratio);
  if (frame->is_pad_resize) {
    w_ratio = resize_ratio;
    h_ratio = resize_ratio;
  }

  // int *shape = tensor->data_shape.d;
  int height, width;
  auto ret =
      HB_BPU_getHW(tensor->data_type, &tensor->data_shape, &height, &width);
  if (ret != 0) {
    LOG(FATAL) << "HB_BPU_getHW failed";
  }

  for (uint32_t h = 0; h < height; h++) {
    for (uint32_t w = 0; w < width; w++) {
      for (int k = 0; k < anchors.size(); k++) {
        double anchor_x = anchors[k].first;
        double anchor_y = anchors[k].second;
        float *cur_data = data + k * num_pred;
        float objness = cur_data[4];
        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] = cur_data[5 + index];
        }

        float id = argmax(class_pred.begin(), class_pred.end());
        double x1 = 1 / (1 + std::exp(-objness)) * 1;
        double x2 = 1 / (1 + std::exp(-class_pred[id]));
        double confidence = x1 * x2;

        if (confidence < score_threshold_) {
          continue;
        }

        float center_x = cur_data[0];
        float center_y = cur_data[1];
        float scale_x = cur_data[2];
        float scale_y = cur_data[3];

        double box_center_x =
            ((1.0 / (1.0 + std::exp(-center_x))) + w) * stride;
        double box_center_y =
            ((1.0 / (1.0 + std::exp(-center_y))) + h) * stride;

        double box_scale_x = std::exp(scale_x) * anchor_x * stride;
        double box_scale_y = std::exp(scale_y) * anchor_y * stride;

        double xmin = (box_center_x - box_scale_x / 2.0);
        double ymin = (box_center_y - box_scale_y / 2.0);
        double xmax = (box_center_x + box_scale_x / 2.0);
        double ymax = (box_center_y + box_scale_y / 2.0);

        double w_padding =
            (frame->width() - w_ratio * frame->ori_width()) / 2.0;
        double h_padding =
            (frame->height() - h_ratio * frame->ori_height()) / 2.0;

        double xmin_org = (xmin - w_padding) / w_ratio;
        double xmax_org = (xmax - w_padding) / w_ratio;
        double ymin_org = (ymin - h_padding) / h_ratio;
        double ymax_org = (ymax - h_padding) / h_ratio;

        if (xmin_org > xmax_org || ymin_org > ymax_org) {
          continue;
        }

        xmin_org = std::max(xmin_org, 0.0);
        xmax_org = std::min(xmax_org, frame->ori_image_width - 1.0);
        ymin_org = std::max(ymin_org, 0.0);
        ymax_org = std::min(ymax_org, frame->ori_image_height - 1.0);

        Bbox bbox(xmin_org, ymin_org, xmax_org, ymax_org);
        dets.push_back(Detection((int)id,
                                 confidence,
                                 bbox,
                                 yolo3_config_.class_names[(int)id].c_str()));
      }
      data = data + num_pred * anchors.size();
    }
  }
}

int Yolo3PostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                        ImageTensor *image_tensor,
                                        Perception *perception) {
  perception->type = Perception::DET;
  std::vector<Detection> dets;
  for (int i = 0; i < yolo3_config_.strides.size(); i++) {
    PostProcess(&tensor[i], image_tensor, i, dets);
  }
  nms(dets, nms_threshold_, nms_top_k_, perception->det, false);
  return 0;
}

int Yolo3PostProcessModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (document.HasMember("nms_threshold")) {
    nms_threshold_ = document["nms_threshold"].GetFloat();
  }

  if (document.HasMember("nms_top_k")) {
    nms_top_k_ = document["nms_top_k"].GetFloat();
  }

  if (document.HasMember("yolo3")) {
    rapidjson::Value &yolo = document["yolo3"];

    // Strides
    rapidjson::Value &stride_value = yolo["strides"];
    auto strides_array = stride_value.GetArray();
    for (int i = 0; i < strides_array.Size(); ++i) {
      yolo3_config_.strides.push_back(strides_array[i].GetInt());
    }

    // Anchors table
    rapidjson::Value &anchors_value = yolo["anchors_table"];
    auto anchors_array = anchors_value.GetArray();
    LOG_IF(FATAL, anchors_array.Size() != yolo3_config_.strides.size())
        << "Feature num should be equal to output num";

    yolo3_config_.anchors_table.reserve(anchors_array.Size());
    for (int i = 0; i < anchors_array.Size(); ++i) {
      auto anchor_pairs = anchors_array[i].GetArray();
      auto &anchors = yolo3_config_.anchors_table[i];
      anchors.reserve(anchor_pairs.Size());

      LOG_IF(FATAL, anchors.size() != yolo3_config_.anchors_table[0].size())
          << "Anchors num should be equal in every feature";

      for (int j = 0; j < anchor_pairs.Size(); ++j) {
        auto anchor = anchor_pairs[j].GetArray();
        anchors[j] =
            std::make_pair(anchor[0].GetDouble(), anchor[1].GetDouble());
      }
    }
    // Class number
    yolo3_config_.class_num = yolo["class_num"].GetInt();

    // Class names
    auto class_names_arr = yolo["class_names"].GetArray();
    yolo3_config_.class_names.resize(yolo3_config_.class_num);
    for (int i = 0; i < yolo3_config_.class_names.size(); i++) {
      yolo3_config_.class_names[i] = class_names_arr[i].GetString();
    }
  }

  return 0;
}
