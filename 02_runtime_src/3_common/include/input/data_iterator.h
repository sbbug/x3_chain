// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_DATA_ITERATOR_H
#define _INPUT_DATA_ITERATOR_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "input/input_data.h"

class DataIterator {
 public:
  explicit DataIterator(std::string module_name) : module_name_(module_name) {}

  /**
   * Init Data iterator from file
   * @param[in] config_file: config file path
   *            the config file should be in json format
   * @param[in] config_string: config string
   *            same as config_file
   * @return 0 if success
   */
  virtual int Init(std::string config_file, std::string config_string);

  /**
   * Next Data
   * @param[out] image_tensor: image tensor
   * @return 0 if success
   */
  virtual bool Next(ImageTensor *image_tensor) = 0;
   /**

  support mutil modal get
  */
  virtual bool Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor) = 0;
  /**
   * Release image_tensor
   * @param[in] image_tensor: image tensor to be released
   */
  virtual void Release(ImageTensor *image_tensor) = 0;

  /**
   * Check if has next input data
   * @return 0 if has next input data
   */
  virtual bool HasNext() = 0;

  /**
   * Get next frame id
   * @return next frame id
   */
  virtual int NextFrameId() { return ++last_frame_id; }

  /**
   * Get DataIterator Implementation instance
   * @param[in]: module_name
   * @return DataIterator implementation instance
   */
  static DataIterator *GetImpl(const std::string &module_name);

  virtual ~DataIterator() {}

 protected:
  virtual int LoadConfig(std::string &config_string) { return 0; }

 private:
  int LoadConfigFile(std::string &config_file);

 private:
  std::string module_name_;

 protected:
  bool is_finish_ = false;
  int last_frame_id = -1;
};

#endif  // _INPUT_DATA_ITERATOR_H
