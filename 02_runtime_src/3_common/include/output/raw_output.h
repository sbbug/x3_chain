// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _OUTPUT_RAW_OUTPUT_H_
#define _OUTPUT_RAW_OUTPUT_H_

#include <fstream>
#include <string>

#include "base/perception_common.h"
#include "output.h"

class RawOutputModule : public OutputModule {
 public:
  RawOutputModule() : OutputModule("raw_output") {}

  /**
   * Init RawOutputModule
   * @param[in] config_file: config file path
   *        config file should be in the json format
   *        for example:
   *        {
   *            "output_file": "raw_output.txt"
   *        }
   * @param[in] config string: config string
   *        same as config file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Write perception data to file
   * @param[in] frame: frame info
   * @param[in] perception: perception data
   */
  void Write(ImageTensor *frame, Perception *perception);

  ~RawOutputModule();

 private:
  int LoadConfig(std::string &config_string);

 private:
  std::string output_file_ = "raw_output.txt";
  std::ofstream ofs_;
};

#endif  // _OUTPUT_RAW_OUTPUT_H_
