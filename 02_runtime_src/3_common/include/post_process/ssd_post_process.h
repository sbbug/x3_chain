// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_SSD_POST_PROCESS_H_
#define _POST_PROCESS_SSD_POST_PROCESS_H_

#include <string>
#include <utility>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "post_process.h"

/**
 * Config definition for SSD
 */
struct SsdConfig {
  std::vector<float> std;
  std::vector<float> mean;
  std::vector<float> offset;
  std::vector<int> step;
  std::vector<std::pair<float, float>> anchor_size;
  std::vector<std::vector<float>> anchor_ratio;
  int class_num;
  std::vector<std::string> class_names;
};

/**
 * Default ssd config
 * std: [0.1, 0.1, 0.2, 0.2]
 * mean: [0, 0, 0, 0]
 * offset: [0.5, 0.5]
 * step: [8, 16, 32, 64, 100, 300]
 * anchor_size: [[30, 60], [60, 111], [111, 162], [162, 213], [213, 264],
 *              [264,315]]
 * anchor_ratio: [[2, 0.5, 0, 0], [2, 0.5, 3, 1.0 / 3],
 *              [2, 0.5,3, 1.0 / 3], [2, 0.5, 3, 1.0 / 3],
 *              [2, 0.5, 0, 0], [2,0.5, 0, 0]]
 * class_num: 20
 * class_names: ["aeroplane",   "bicycle", "bird",  "boaupdate", "bottle",
     "bus",         "car",     "cat",   "chair",     "cow",
     "diningtable", "dog",     "horse", "motorbike", "person",
     "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"]
 */
extern SsdConfig default_ssd_config;

class SsdPostProcessModule : public PostProcessModule {
 public:
  explicit SsdPostProcessModule(std::string instance_name)
      : PostProcessModule("ssd_post_process", instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "score_threshold": 0.2,
   *        "nms_threshold": 0.2,
   *        "ssd": {
   *            "std": ...
   *            "mean": ...
   *            "offset": ...
   *            "step": ...
   *            ...
   *        }
   *    }
   * @param[in] config_string: config string
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Post process
   * @param[in] tensor: Model output tensors
   * @param[in] image_tensor: Input image tensor
   * @param[out] perception: Perception output data
   * @return 0 if success
   */
  int PostProcess(BPU_TENSOR_S *tensor,
                  ImageTensor *image_tensor,
                  Perception *perception);

 private:
  int LoadConfig(std::string &config_string);

  int SoftmaxFromRawScore(BPU_TENSOR_S *tensor,
                          int class_num,
                          std::vector<double> &scores,
                          float cut_off_threshold);

  int GetBboxFromRawData(BPU_TENSOR_S *tensor,
                         std::vector<Anchor> &anchors,
                         std::vector<Bbox> &bboxes,
                         ImageTensor *frame);

  int SsdAnchors(std::vector<Anchor> &anchors,
                 int layer,
                 int layer_height,
                 int layer_width);

 private:
  SsdConfig ssd_config_ = default_ssd_config;
  float score_threshold_ = 0.3;
  float nms_threshold_ = 0.3;
  int nms_top_k_ = 200;
};

#endif  // _POST_PROCESS_SSD_POST_PROCESS_H_
