// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _OUTPUT_VIDEO_OUTPUT_H_
#define _OUTPUT_VIDEO_OUTPUT_H_

#include <string>

#include "base/perception_common.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "output.h"

class VideoOutputModule : public OutputModule {
 public:
  VideoOutputModule() : OutputModule("video_output") {}

  /**
   * Init VideoOutputModule from config file
   * @param[in] config_file: config file path
   *        config file should be in the json format
   *        for example:
   *        {
   *            "video_name": "output.avi",
   *            "fps": 25,
   *            "height": 480,
   *            "width": 640
   *         }
   * @param[in] config_string: config string
   *        same as config file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Draw perception data on image then write it to video file
   * @param[in] frame: frame info
   * @param[in] perception: perception data
   */
  void Write(ImageTensor *frame, Perception *perception);

  ~VideoOutputModule();

 private:
  int LoadConfig(std::string &config_string);

  cv::VideoWriter video_writer_;
  int fps_ = 25;
  int height_ = 480;
  int width_ = 640;
  std::string video_name_ = "output.avi";
};

#endif  // _OUTPUT_VIDEO_OUTPUT_H_
