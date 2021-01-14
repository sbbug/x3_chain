// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _OUTPUT_OUTPUT_H_
#define _OUTPUT_OUTPUT_H_

#include <string>

#include "base/perception_common.h"
#include "input/input_data.h"

class OutputModule {
 public:
  explicit OutputModule(std::string module_name) : module_name_(module_name) {}

  /**
   * Init OutputModule from file
   * @param[in] config_file: config file path
   * @return 0 if success
   */
  virtual int Init(std::string config_file, std::string config_string);

  /**
   * Write perception data to output
   * @param[in] frame: frame info
   * @param[in] perception: perception data
   */
  virtual void Write(ImageTensor* frame, Perception* perception) = 0;

  /**
   * Get OutputModule Implementation instance
   * @param[in]: module_name
   * @return OutputModule implementation instance
   */
  static OutputModule* GetImpl(const std::string& module_name);

  virtual ~OutputModule() {}

 protected:
  int LoadConfigFile(std::string& config_file);

  virtual int LoadConfig(std::string& config_string) { return 0; }

 private:
  std::string module_name_;
};

#endif  // _OUTPUT_OUTPUT_H_
