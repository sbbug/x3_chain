// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_S3FD_POST_PROCESS_H_
#define _POST_PROCESS_S3FD_POST_PROCESS_H_

#include <string>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "post_process.h"

struct S3fdConfig {
  std::vector<float> variance;
  std::vector<int> step;
  std::vector<std::pair<int, int>> min_size;
  int class_num;
};

/**
 * Default s3fd config
 * variance: [0.1, 0.2]
 * step: [4, 8, 16, 32, 64, 128]
 * min_size: [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]
 * class_num: 1
 */
extern S3fdConfig default_s3fd_config;

class S3fdPostProcessModule : public PostProcessModule {
 public:
  explicit S3fdPostProcessModule(std::string instance_name)
      : PostProcessModule("s3fd_post_process", instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "score_threshold": 0.2,
   *        "nms_threshold_": 0.2,
   *        "nms_top_k": 750,
   *        "s3fd": {
   *            "variance": ...
   *            "step": ...
   *            "min_size": ...
   *            "class_num": ...
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

  int GetBboxFromRawData(BPU_TENSOR_S *tensor,
                         std::vector<Anchor> &anchors,
                         std::vector<Bbox> &bboxes);

  int SoftmaxFromRawScore(BPU_TENSOR_S *tensor,
                          std::vector<double> &scores,
                          float cut_off_threshold);

  int S3fdAnchors(std::vector<Anchor> &anchor_table,
                  int layer,
                  int layer_height,
                  int layer_width);

 private:
  S3fdConfig s3fd_config_ = default_s3fd_config;
  float score_threshold_ = 0.2;
  float nms_threshold_ = 0.2;
  int nms_top_k_ = 750;
};

#endif  // _POST_PROCESS_S3FD_POST_PROCESS_H_
