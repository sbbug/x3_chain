// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/ssd_post_process.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/nms.h"

#define SSD_CLASS_NUM_P1 21

SsdConfig default_ssd_config = {
    {0.1, 0.1, 0.2, 0.2},
    {0, 0, 0, 0},
    {0.5, 0.5},
    {8, 16, 32, 64, 100, 300},
    {{30, 60}, {60, 111}, {111, 162}, {162, 213}, {213, 264}, {264, 315}},
    {{2, 0.5, 0, 0},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 3, 1.0 / 3},
     {2, 0.5, 0, 0},
     {2, 0.5, 0, 0}},
    20,
    {"aeroplane",   "bicycle", "bird",  "boaupdate", "bottle",
     "bus",         "car",     "cat",   "chair",     "cow",
     "diningtable", "dog",     "horse", "motorbike", "person",
     "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"}};

int SsdPostProcessModule::Init(std::string config_file,
                               std::string config_string) {
  DLOG(INFO) << "Init " << FullName();
  int ret_code = PostProcessModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

int SsdPostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                      ImageTensor *image_tensor,
                                      Perception *perception) {
  perception->type = Perception::DET;
  std::vector<std::vector<Anchor>> anchors_table;
  int layer_num = ssd_config_.step.size();

  anchors_table.resize(layer_num);
  for (int i = 0; i < layer_num; i++) {
    int height, width;
    HB_BPU_getHW(
        tensor[i * 2].data_type, &tensor[i * 2].aligned_shape, &height, &width);
    // int *shape = tensor[i * 2].data_shape.d;
    SsdAnchors(anchors_table[i], i, height, width);
  }
  std::vector<Detection> dets;
  for (int i = 0; i < layer_num; i++) {
    std::vector<Anchor> &anchors = anchors_table[i];
    int anchors_num = anchors.size();
    std::vector<double> cls_scores(SSD_CLASS_NUM_P1 * anchors_num);
    SoftmaxFromRawScore(&tensor[i * 2 + 1], SSD_CLASS_NUM_P1, cls_scores, 0.01);
    std::vector<Bbox> bboxes(anchors_num);
    GetBboxFromRawData(&tensor[i * 2], anchors, bboxes, image_tensor);
    for (int j = 0; j < anchors_num; j++) {
      Bbox &bbox = bboxes[j];
      auto cls_score = cls_scores.data() + j * SSD_CLASS_NUM_P1;
      for (int w = 1; w < SSD_CLASS_NUM_P1; w++) {
        if (cls_score[w] > score_threshold_) {
          dets.push_back(Detection(w - 1,
                                   cls_score[w],
                                   bbox,
                                   ssd_config_.class_names[w - 1].c_str()));
        }
      }
    }
  }

  nms(dets, nms_threshold_, nms_top_k_, perception->det, false);
  return 0;
}

int SsdPostProcessModule::GetBboxFromRawData(BPU_TENSOR_S *tensor,
                                             std::vector<Anchor> &anchors,
                                             std::vector<Bbox> &bboxes,
                                             ImageTensor *frame) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *raw_box_data = reinterpret_cast<float *>(tensor->data.virAddr);

  int *shape = tensor->data_shape.d;

  int32_t batch_size = shape[0];

  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(
      tensor->data_type, &tensor->aligned_shape.layout, &h_idx, &w_idx, &c_idx);

  int32_t hnum = shape[h_idx];
  int32_t wnum = shape[w_idx];
  int32_t cnum = shape[c_idx];

  float scale_h = 1.0 * frame->ori_height() / frame->height();
  float scale_w = 1.0 * frame->ori_width() / frame->width();

  uint32_t w_dest_stride = cnum / 4;
  uint32_t h_dest_stride = wnum * w_dest_stride;
  uint32_t anchor_num_per_pixel = cnum / 4;
  for (uint32_t hh = 0; hh < batch_size * hnum; hh++) {
    uint32_t res_id_cur_hh = hh * h_dest_stride;
    for (uint32_t ww = 0; ww < wnum; ww++) {
      uint32_t res_id_cur_ww = res_id_cur_hh + ww * w_dest_stride;
      for (uint32_t anchor_id = 0; anchor_id < anchor_num_per_pixel;
           anchor_id++) {
        uint32_t res_id_cur_anchor = res_id_cur_ww + anchor_id;
        uint32_t cur_c = 4 * anchor_id;
        float *box = raw_box_data + hh * wnum * cnum + ww * cnum + cur_c;
        float box_0 = box[0];
        float box_1 = box[1];
        float box_2 = box[2];
        float box_3 = box[3];
        float ox = ((box_0 * ssd_config_.std[0] + ssd_config_.mean[0]) *
                    anchors[res_id_cur_anchor].w) +
                   anchors[res_id_cur_anchor].cx;
        float oy = ((box_1 * ssd_config_.std[1] + ssd_config_.mean[1]) *
                    anchors[res_id_cur_anchor].h) +
                   anchors[res_id_cur_anchor].cy;
        float tw, th;
        tw = std::exp(box_2 * ssd_config_.std[2] + ssd_config_.mean[2]);
        th = std::exp(box_3 * ssd_config_.std[3] + ssd_config_.mean[3]);
        float ow = (tw * anchors[res_id_cur_anchor].w) / 2.0;
        float oh = (th * anchors[res_id_cur_anchor].h) / 2.0;
        bboxes[res_id_cur_anchor].xmin = (ox - ow) * scale_w;
        bboxes[res_id_cur_anchor].ymin = (oy - oh) * scale_h;
        bboxes[res_id_cur_anchor].xmax = (ox + ow) * scale_w;
        bboxes[res_id_cur_anchor].ymax = (oy + oh) * scale_h;
      }
    }
  }
  return 0;
}

int SsdPostProcessModule::SoftmaxFromRawScore(BPU_TENSOR_S *tensor,
                                              int class_num,
                                              std::vector<double> &scores,
                                              float cut_off_threshold) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *raw_cls_data = reinterpret_cast<float *>(tensor->data.virAddr);

  int *shape = tensor->data_shape.d;
  int32_t batch_size = shape[0];
  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(
      tensor->data_type, &tensor->aligned_shape.layout, &h_idx, &w_idx, &c_idx);
  int32_t hnum = shape[h_idx];
  int32_t wnum = shape[w_idx];
  int32_t cnum = shape[c_idx];

  uint32_t w_dest_stride = cnum;
  uint32_t h_dest_stride = wnum * w_dest_stride;
  uint32_t anchor_num_per_pixel = cnum / class_num;
  for (uint32_t hh = 0; hh < batch_size * hnum; hh++) {
    uint32_t res_id_cur_hh = hh * h_dest_stride;
    for (uint32_t ww = 0; ww < wnum; ww++) {
      uint32_t res_id_cur_ww = res_id_cur_hh + ww * w_dest_stride;
      for (uint32_t anchor_id = 0; anchor_id < anchor_num_per_pixel;
           anchor_id++) {
        uint32_t res_id_cur_anchor = res_id_cur_ww + anchor_id * class_num;
        double sum = 0;
        for (int cls = 0; cls < class_num; ++cls) {
          float cls_value = raw_cls_data[res_id_cur_anchor + cls];
          scores[res_id_cur_anchor + cls] = std::exp(cls_value);
          sum += scores[res_id_cur_anchor + cls];
        }
        for (int cls = 0; cls < class_num; ++cls) {
          scores[res_id_cur_anchor + cls] =
              scores[res_id_cur_anchor + cls] / sum;
          if (scores[res_id_cur_anchor + cls] < cut_off_threshold) {
            scores[res_id_cur_anchor + cls] = 0;
          }
        }
      }
    }
  }
  return 0;
}

int SsdPostProcessModule::SsdAnchors(std::vector<Anchor> &anchors,
                                     int layer,
                                     int layer_height,
                                     int layer_width) {
  int step = ssd_config_.step[layer];
  float min_size = ssd_config_.anchor_size[layer].first;
  float max_size = ssd_config_.anchor_size[layer].second;
  auto &anchor_ratio = ssd_config_.anchor_ratio[layer];
  for (int i = 0; i < layer_height; i++) {
    for (int j = 0; j < layer_width; j++) {
      float cy = (i + ssd_config_.offset[0]) * step;
      float cx = (j + ssd_config_.offset[1]) * step;
      anchors.emplace_back(Anchor(cx, cy, min_size, min_size));
      anchors.emplace_back(Anchor(cx,
                                  cy,
                                  std::sqrt(max_size * min_size),
                                  std::sqrt(max_size * min_size)));
      for (int k = 0; k < 4; k++) {
        if (anchor_ratio[k] == 0) continue;
        float sr = std::sqrt(anchor_ratio[k]);
        float w = min_size * sr;
        float h = min_size / sr;
        anchors.emplace_back(Anchor(cx, cy, w, h));
      }
    }
  }
  return 0;
}

int SsdPostProcessModule::LoadConfig(std::string &config_string) {
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

  if (document.HasMember("ssd")) {
    rapidjson::Value &ssd = document["ssd"];

    // Std
    auto std_array = ssd["std"].GetArray();
    ssd_config_.std.resize(std_array.Size());
    for (int i = 0; i < std_array.Size(); i++) {
      ssd_config_.std[i] = std_array[i].GetFloat();
    }

    // Mean
    auto mean_array = ssd["mean"].GetArray();
    ssd_config_.mean.resize(mean_array.Size());
    for (int i = 0; i < mean_array.Size(); i++) {
      ssd_config_.mean[i] = mean_array[i].GetFloat();
    }

    // Offset
    auto offset_array = ssd["offset"].GetArray();
    ssd_config_.offset.resize(offset_array.Size());
    for (int i = 0; i < offset_array.Size(); i++) {
      ssd_config_.offset[i] = offset_array[i].GetFloat();
    }

    // Step
    auto step_array = ssd["step"].GetArray();
    ssd_config_.step.resize(step_array.Size());
    for (int i = 0; i < step_array.Size(); i++) {
      ssd_config_.step[i] = step_array[i].GetInt();
    }

    // Anchor size
    auto anchor_size_array = ssd["anchor_size"].GetArray();
    ssd_config_.anchor_size.resize(anchor_size_array.Size() / 2);
    for (int i = 0; i < anchor_size_array.Size(); i += 2) {
      ssd_config_.anchor_size[i] = std::make_pair(
          anchor_size_array[i].GetFloat(), anchor_size_array[i + 1].GetFloat());
    }

    // Anchor ratio
    auto ratio_array = ssd["anchor_ratio"].GetArray();
    ssd_config_.anchor_ratio.resize(ratio_array.Size() / 4);
    for (int i = 0; i < ratio_array.Size(); i += 4) {
      std::vector<float> &ratio = ssd_config_.anchor_ratio[i];
      ratio[0] = ratio_array[i].GetFloat();
      ratio[1] = ratio_array[i + 1].GetFloat();
      ratio[2] = ratio_array[i + 2].GetFloat();
      ratio[3] = ratio_array[i + 3].GetFloat();
    }

    // Class num
    ssd_config_.class_num = ssd["class_num"].GetInt();

    // Class names
    auto class_names_arr = ssd["class_names"].GetArray();
    ssd_config_.class_names.resize(ssd_config_.class_num);
    for (int i = 0; i < ssd_config_.class_names.size(); i++) {
      ssd_config_.class_names[i] = class_names_arr[i].GetString();
    }

    return 0;
  }
}
