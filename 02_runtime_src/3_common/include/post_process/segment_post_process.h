// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_SEGMENT_POST_PROCESS_H_
#define _POST_PROCESS_SEGMENT_POST_PROCESS_H_

#include <string>
#include <vector>

#include "bpu_predict_extension.h"
#include "post_process.h"

class SegmentPostProcessModule : public PostProcessModule {
 public:
  explicit SegmentPostProcessModule(std::string instance_name)
      : PostProcessModule("segment_post_process", instance_name) {}

  /**
   * Load configuration from file
   * Note: Here we do not need config file for Segment post process
   *       Just keep api consistent with  `PostProcessModule`
   * @param[in] config_file: config file path
   * @param[in] config_string: config string
   */
  int Init(std::string config_file, std::string config_string) { return 0; }

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
};

#endif  // _POST_PROCESS_SEGMENT_POST_PROCESS_H_
