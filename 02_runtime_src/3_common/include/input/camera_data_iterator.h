// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_CAMERA_DATA_ITERATOR_H_
#define _INPUT_CAMERA_DATA_ITERATOR_H_

#include <string>

#include "data_iterator.h"

class CameraDataIterator : public DataIterator {
 public:
  CameraDataIterator() : DataIterator("camera_data_iterator") {}

  /**
   * Init Camera data iterator
   * @param[in] config_file: config file
   *        the file content should be in the json format
   *        for example:
   *        {
   *            "cam_cfg" : "hb_camera_720P.json",
   *            "vio_cfg": "hb_vio_720P.json",
   *            "cam_index": 0,
   *            "cam_port": 0,
   *            "frame_count": -1 # -1 represent infinite
   *        }
   * @param[in] config_string: config string
   *        same as config_file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Next Image Data read from file system
   * @param[out] image_tensor: image tensor
   * @return 0 if success
   */
  bool Next(ImageTensor *image_tensor);
   /**

  support mutil modal get
  */
  bool Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor);
  /**
   * Release image_tensor
   * @param[in] image_tensor: image tensor to be released
   */
  void Release(ImageTensor *image_tensor);

  /**
   * Check if has next image
   * @return false if finish
   */
  bool HasNext();

  ~CameraDataIterator() {}

 private:
  int LoadConfig(std::string &config_string);

 private:
  BPUCameraHandle cam_handle_;
  std::string cam_cfg_ = "";
  std::string vio_cfg_ = "";
  int cam_index_ = 0;
  int cam_port_ = 0;
  int frame_count_ = -1;
  int count_ = 0;
};

#endif  // _INPUT_CAMERA_DATA_ITERATOR_H_
