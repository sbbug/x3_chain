// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_CLASSIFICATION_POST_PROCESS_H_
#define _POST_PROCESS_CLASSIFICATION_POST_PROCESS_H_

#include <string>
#include <utility>
#include <vector>

#include "bpu_predict_extension.h"
#include "post_process.h"

/**
 * Classification post process
 * Note: Only float32 output support here for now
 */
class ClassificationPostProcessModule : public PostProcessModule {
 public:
  explicit ClassificationPostProcessModule(std::string instance_name)
      : PostProcessModule("classification_post_process", instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "top_k" :5,
   *        "class_names": []
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

  ~ClassificationPostProcessModule() = default;

 private:
  int LoadConfig(std::string &config_string);

  void GetMaxResult(BPU_TENSOR_S *tensor, Classification *cls);

  void GetTopkResult(BPU_TENSOR_S *tensor,
                     std::vector<Classification> &top_k_cls);

  const char *GetClsName(int id);

 private:
  int top_k_ = 1;
  std::vector<std::string> class_names_;
};

#endif  // _POST_PROCESS_CLASSIFICATION_POST_PROCESS_H_
