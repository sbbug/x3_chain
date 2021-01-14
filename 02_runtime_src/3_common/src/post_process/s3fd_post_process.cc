// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "post_process/s3fd_post_process.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/nms.h"

S3fdConfig default_s3fd_config = {
    {0.1, 0.2},
    {4, 8, 16, 32, 64, 128},
    {{16, 16}, {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512}},
    1};

int S3fdPostProcessModule::Init(std::string config_file,
                                std::string config_string) {
  DLOG(INFO) << "Init " << FullName();
  int ret_code = PostProcessModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

int S3fdPostProcessModule::PostProcess(BPU_TENSOR_S *tensor,
                                       ImageTensor *image_tensor,
                                       Perception *perception) {
  perception->type = Perception::DET;
  std::vector<std::vector<Anchor>> anchors_table;
  int layer_num = s3fd_config_.step.size();

  anchors_table.resize(layer_num);
  for (int i = 0; i < layer_num; i++) {
    int height, width;
    HB_BPU_getHW(
        tensor[i * 2].data_type, &tensor[i * 2].aligned_shape, &height, &width);
    // int *shape = tensor[i * 2].aligned_shape.d;
    S3fdAnchors(anchors_table[i], i, height, width);
  }
  std::vector<Detection> dets;
  for (int i = 0; i < layer_num; i++) {
    std::vector<Anchor> &anchors = anchors_table[i];
    int anchors_num = anchors.size();
    std::vector<double> cls_scores(s3fd_config_.class_num * anchors_num);
    SoftmaxFromRawScore(&tensor[i * 2 + 1], cls_scores, 0.01);
    std::vector<Bbox> bboxes(anchors_num);
    GetBboxFromRawData(&tensor[i * 2], anchors, bboxes);

    for (int j = 0; j < anchors_num; j++) {
      Bbox bbox = bboxes[j];
      auto cls_all = cls_scores[j];
      if (cls_all >= score_threshold_) {
        bbox.xmin *= (image_tensor->ori_width() / image_tensor->width());
        bbox.ymin *= (image_tensor->ori_height() / image_tensor->height());
        bbox.xmax *= (image_tensor->ori_width() / image_tensor->width());
        bbox.ymax *= (image_tensor->ori_height() / image_tensor->height());
        dets.push_back(Detection(0, cls_all, bbox));
      }
    }
  }

  nms(dets, nms_threshold_, nms_top_k_, perception->det, false);
  return 0;
}

int S3fdPostProcessModule::GetBboxFromRawData(BPU_TENSOR_S *tensor,
                                              std::vector<Anchor> &anchors,
                                              std::vector<Bbox> &bboxes) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *raw_box_data = reinterpret_cast<float *>(tensor->data.virAddr);

  int *shape = tensor->aligned_shape.d;

  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(
      tensor->data_type, &tensor->aligned_shape.layout, &h_idx, &w_idx, &c_idx);

  int32_t hnum = shape[h_idx];
  int32_t wnum = shape[w_idx];
  int32_t cnum = shape[c_idx];

  uint32_t h_dest_stride = wnum;
  for (uint32_t hh = 0; hh < hnum; hh++) {
    uint32_t res_id_cur_hh = hh * h_dest_stride;
    for (uint32_t ww = 0; ww < wnum; ww++) {
      uint32_t res_id_cur_anchor = res_id_cur_hh + ww;
      float *box = raw_box_data + hh * wnum * cnum + ww * cnum;
      float box_0 = box[0];
      float box_1 = box[1];
      float box_2 = box[2];
      float box_3 = box[3];
      float ox =
          anchors[res_id_cur_anchor].cx +
          box_0 * s3fd_config_.variance[0] * anchors[res_id_cur_anchor].w;
      float oy =
          anchors[res_id_cur_anchor].cy +
          box_1 * s3fd_config_.variance[0] * anchors[res_id_cur_anchor].h;
      float tw = anchors[res_id_cur_anchor].w *
                 std::exp(box_2 * s3fd_config_.variance[1]);
      float th = anchors[res_id_cur_anchor].h *
                 std::exp(box_3 * s3fd_config_.variance[1]);
      float ow = tw / 2;
      float oh = th / 2;
      bboxes[res_id_cur_anchor].xmin = ox - ow;
      bboxes[res_id_cur_anchor].ymin = oy - oh;
      bboxes[res_id_cur_anchor].xmax = ox + ow;
      bboxes[res_id_cur_anchor].ymax = oy + oh;
    }
  }
  return 0;
}

int S3fdPostProcessModule::SoftmaxFromRawScore(BPU_TENSOR_S *tensor,
                                               std::vector<double> &scores,
                                               float cut_off_threshold) {
  HB_SYS_flushMemCache(&(tensor->data), HB_SYS_MEM_CACHE_INVALIDATE);
  auto *raw_cls_data = reinterpret_cast<float *>(tensor->data.virAddr);

  int *shape = tensor->aligned_shape.d;

  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(
      tensor->data_type, &tensor->aligned_shape.layout, &h_idx, &w_idx, &c_idx);

  int32_t hnum = shape[h_idx];
  int32_t wnum = shape[w_idx];
  int32_t cnum = shape[c_idx];

  for (uint32_t hh = 0; hh < hnum; hh++) {
    uint32_t res_id_cur_hh = hh * wnum;
    for (uint32_t ww = 0; ww < wnum; ww++) {
      uint32_t res_id_cur_anchor = res_id_cur_hh + ww;
      float c_score[2];
      for (int cls = 0; cls < cnum; ++cls) {
        float conv_res_f = raw_cls_data[hh * wnum * cnum + ww * cnum + cls];

        if (cls == cnum - 1) {
          c_score[1] = conv_res_f;
          continue;
        }

        if (cls == 0) {
          c_score[0] = conv_res_f;
          continue;
        }

        if (c_score[0] < conv_res_f) {
          c_score[0] = conv_res_f;
        }
      }

      float mmax = std::fmax(c_score[0], c_score[1]);
      for (float &cls : c_score) {
        cls = std::exp(cls - mmax);
      }

      float sum = c_score[0] + c_score[1];
      // softmax
      float cur_score = c_score[1] / sum;
      scores[res_id_cur_anchor] =
          cur_score >= cut_off_threshold ? cur_score : 0;
    }
  }
  return 0;
}

int S3fdPostProcessModule::S3fdAnchors(std::vector<Anchor> &anchor_table,
                                       int layer,
                                       int layer_height,
                                       int layer_width) {
  std::vector<Anchor> anchors;
  auto &min_size = s3fd_config_.min_size[layer];
  auto step = s3fd_config_.step[layer];
  for (int i = 0; i < layer_height; i++) {
    for (int j = 0; j < layer_width; j++) {
      float s_kx = min_size.second;
      float s_ky = min_size.first;
      float cx = (j + 0.5) * step;
      float cy = (i + 0.5) * step;
      anchors.push_back(Anchor(cx, cy, s_kx, s_ky));
    }
  }
  return 0;
}

int S3fdPostProcessModule::LoadConfig(std::string &config_string) {
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

  if (document.HasMember("s3fd")) {
    rapidjson::Value &s3fd = document["s3fd"];

    // Variance
    auto variance_array = s3fd["variance"].GetArray();
    s3fd_config_.variance.resize(variance_array.Size());
    for (auto i = 0; i < variance_array.Size(); i++) {
      s3fd_config_.variance[i] = variance_array[i].GetFloat();
    }

    // Step
    auto step_array = s3fd["step"].GetArray();
    s3fd_config_.step.resize(step_array.Size());
    for (int i = 0; i < step_array.Size(); i++) {
      s3fd_config_.step[i] = step_array[i].GetInt();
    }

    // Min size
    auto min_size_array = s3fd["min_size"].GetArray();
    s3fd_config_.min_size.resize(min_size_array.Size() / 2);
    for (int i = 0; i < min_size_array.Size(); i += 2) {
      s3fd_config_.min_size[i] = std::make_pair(min_size_array[i].GetInt(),
                                                min_size_array[i + 1].GetInt());
    }

    // Class num
    s3fd_config_.class_num = s3fd["class_num"].GetInt();
  }

  return 0;
}
