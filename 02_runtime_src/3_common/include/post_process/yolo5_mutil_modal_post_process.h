// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_MUTIL_MODAL_YOLO5_POST_PROCESS_H_
#define _POST_PROCESS_MUTIL_MODAL_YOLO5_POST_PROCESS_H_

#include <string>
#include <utility>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "post_process.h"

/**
 * Config definition for Yolo5
 */
struct Yolo5MutilModalConfig {
  std::vector<int> strides;
  std::vector<std::vector<std::pair<double, double>>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;
};

extern Yolo5MutilModalConfig default_yolo5_mutil_modal_config;

class Yolo5MutilModalPostProcessModule : public PostProcessModule {
 public:
  explicit Yolo5MutilModalPostProcessModule(std::string instance_name)
      : PostProcessModule("yolo5_mutil_modal_post_process", instance_name) {}
  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "score_threshold": 0.3,
   *        "nms_threshold": 0.45,
   *        "nms_top_k": 500,
   *        "yolov5": {
   *            "strides": ...
   *            "anchors_table": ...
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

  void PostProcess(BPU_TENSOR_S *tensor,
                   ImageTensor *frame,
                   int layer,
                   std::vector<Detection> &dets);

 private:
  Yolo5MutilModalConfig yolo5_config_ = default_yolo5_mutil_modal_config;
  float score_threshold_ = 0.001;
  float nms_threshold_ = 0.65;
  int nms_top_k_ = 5000;
};

#endif  // _POST_PROCESS_YOLO5_POST_PROCESS_H_
