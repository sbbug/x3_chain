// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_FASTERRCNN_POST_PROCESS_H_
#define _POST_PROCESS_FASTERRCNN_POST_PROCESS_H_

#include <string>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "post_process.h"

class FasterrcnnPostProcessModule : public PostProcessModule {
 public:
  explicit FasterrcnnPostProcessModule(std::string instance_name)
      : PostProcessModule("fasterrcnn_post_process", instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "score_threshold": 0.6,
   *        "class_num": 20
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
  int PostProcess(BPU_TENSOR_S* tensor,
                  ImageTensor* image_tensor,
                  Perception* perception);

 private:
  int LoadConfig(std::string& config_string);

 private:
  float score_threshold_ = 0.8;
  int class_num_ = 20;
  std::vector<std::string> class_names_{
      "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
      "bus",         "car",     "cat",   "chair",     "cow",
      "diningtable", "dog",     "horse", "motorbike", "person",
      "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"};
};

#endif  // _POST_PROCESS_FASTERRCNN_POST_PROCESS_H_
