// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/yolo2_post_process.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "utils/algorithm.h"
#include "utils/nms.h"

Yolo2Config default_yolo2_config = {
    32,
    {{0.57273, 0.677385},
     {1.87446, 2.06253},
     {3.33843, 5.47434},
     {7.88282, 3.52778},
     {9.77052, 9.16828}},
    80,
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"}};

int Yolo2PostProcessModule::Init(std::string config_file,
                                 std::string config_string) {
  DLOG(INFO) << "Init " << FullName();
  int ret_code = PostProcessModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

int Yolo2PostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                        ImageTensor *image_tensor,
                                        Perception *perception) {
  perception->type = Perception::DET;
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *data = reinterpret_cast<float *>(tensor->data.virAddr);
  auto &anchors_table = yolo2_config_.anchors_table;
  int num_classes = yolo2_config_.class_num;
  float stride = static_cast<float>(yolo2_config_.stride);
  int num_pred = num_classes + 4 + 1;
  std::vector<Detection> dets;
  std::vector<float> class_pred(num_classes, 0.0);
  double w_ratio = image_tensor->width() * 1.0 / image_tensor->ori_width();
  double h_ratio = image_tensor->height() * 1.0 / image_tensor->ori_height();
  double resize_ratio = std::min(w_ratio, h_ratio);
  if (image_tensor->is_pad_resize) {
    w_ratio = resize_ratio;
    h_ratio = resize_ratio;
  }
  int height, width;
  HB_BPU_getHW(tensor->data_type, &tensor->data_shape, &height, &width);
  // int *shape = tensor->data_shape.d;
  for (uint32_t h = 0; h < height; h++) {
    for (uint32_t w = 0; w < width; w++) {
      for (int k = 0; k < anchors_table.size(); k++) {
        double anchor_x = anchors_table[k].first;
        double anchor_y = anchors_table[k].second;
        float *cur_data = data + k * num_pred;

        float objness = cur_data[4];
        for (int index = 0; index < num_classes; ++index) {
          class_pred[index] = cur_data[5 + index];
        }

        float id = argmax(class_pred.begin(), class_pred.end());

        float confidence = (1.f / (1 + std::exp(-objness))) *
                           (1.f / (1 + std::exp(-class_pred[id])));

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
            (image_tensor->width() - w_ratio * image_tensor->ori_width()) / 2.0;
        double h_padding =
            (image_tensor->height() - h_ratio * image_tensor->ori_height()) /
            2.0;

        double xmin_org = (xmin - w_padding) / w_ratio;
        double xmax_org = (xmax - w_padding) / w_ratio;
        double ymin_org = (ymin - h_padding) / h_ratio;
        double ymax_org = (ymax - h_padding) / h_ratio;

        if (xmin_org > xmax_org || ymin_org > ymax_org) {
          continue;
        }

        xmin_org = std::max(xmin_org, 0.0);
        xmax_org = std::min(xmax_org, image_tensor->ori_image_width - 1.0);
        ymin_org = std::max(ymin_org, 0.0);
        ymax_org = std::min(ymax_org, image_tensor->ori_image_height - 1.0);

        Bbox bbox(xmin_org, ymin_org, xmax_org, ymax_org);
        dets.emplace_back(
            Detection((int)id,
                      confidence,
                      bbox,
                      yolo2_config_.class_names[(int)id].c_str()));
      }
      data = data + num_pred * anchors_table.size();
    }
  }

  nms(dets, nms_threshold_, nms_top_k_, perception->det, false);
  return 0;
}

int Yolo2PostProcessModule::LoadConfig(std::string &config_string) {
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

  if (document.HasMember("yolo2")) {
    rapidjson::Value &yolo = document["yolo2"];

    // Strides
    rapidjson::Value &stride_value = yolo["stride"];
    yolo2_config_.stride = yolo["stride"].GetInt();

    // Anchors table
    rapidjson::Value &anchors_value = yolo["anchors_table"];
    auto anchors_array = anchors_value.GetArray();
    yolo2_config_.anchors_table.resize(anchors_array.Size() / 2);
    for (int i = 0; i < anchors_array.Size(); i += 2) {
      yolo2_config_.anchors_table[i] = std::make_pair(
          anchors_array[i].GetDouble(), anchors_array[i + 1].GetDouble());
    }

    // Class num
    yolo2_config_.class_num = yolo["class_num"].GetInt();

    // Class names
    auto class_names_arr = yolo["class_names"].GetArray();
    yolo2_config_.class_names.resize(yolo2_config_.class_num);
    for (int i = 0; i < yolo2_config_.class_names.size(); i++) {
      yolo2_config_.class_names[i] = class_names_arr[i].GetString();
    }
  }

  return 0;
}
