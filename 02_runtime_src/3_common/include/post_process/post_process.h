// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include <string>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "input/input_data.h"
#include "utils/stop_watch.h"

#ifndef _POST_PROCESS_POST_PROCESS_H_
#define _POST_PROCESS_POST_PROCESS_H_

class PostProcessModule {
 public:
  explicit PostProcessModule(std::string module_name, std::string instance_name)
      : module_name_(module_name), instance_name_(instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   * @param[in] config_string: config string
   * @return 0 if success
   */
  virtual int Init(std::string config_file, std::string config_string);

  /**
   * Post process
   * @param[in] tensor: Model output tensors
   * @param[in] image_tensor: Input image tensor
   * @param[out] perception: Perception output data
   * @return 0 if success
   */
  virtual int PostProcess(BPU_TENSOR_S *tensor,
                          ImageTensor *image_tensor,
                          Perception *perception) = 0;

  /**
   * Get Full name
   * @return  full name (module_name:instance_name)
   */
  std::string FullName();

  /**
   * Get PostProcess Implementation instance
   * @param model_name
   * @return PostProcess implementation instance
   */
  static PostProcessModule *GetImpl(const std::string &model_name);

  virtual ~PostProcessModule() = default;

 protected:
  virtual int LoadConfig(std::string &config_string) { return 0; }

 private:
  int LoadConfigFile(std::string &config_file);

 protected:
  std::string module_name_;
  std::string instance_name_;
  Stopwatch stop_watch_;
};

#endif  // _POST_PROCESS_POST_PROCESS_H_
