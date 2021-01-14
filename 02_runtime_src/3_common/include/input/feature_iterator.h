// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_FEATURE_ITERATOR_H_
#define _INPUT_FEATURE_ITERATOR_H_

#include <string>
#include <vector>

#include "data_iterator.h"

class FeatureIterator : public DataIterator {
 public:
  FeatureIterator() : DataIterator("feature_iterator") {}

  /**
   * Init feature iterator from file
   * @param[in] config_file: config file
   *        config file should be in the json format
   *        for example:
   *        {
   *            "feature_list_file": "feature_list.txt",
   *            "data_type": 9,
   *        }
   *        the `feature_list_file` should be one data file per line
   *        and the data file name should be in for format of
   *        {name}_{dim0}_{dim1}_{dim2}_{...}.bin
   * @param[in] config_string: config string
   *        same as config file
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
   * @return 0 if finish
   */
  bool HasNext();

  ~FeatureIterator();

 private:
  int LoadConfig(std::string &config_string);

  void ParsePathParams(std::string &input_file,
                       std::string &image_name,
                       std::vector<int> &dims);

 private:
  std::string feature_list_file_;
  std::ifstream ifs_;
  hb_BPU_DATA_TYPE_E data_type_ = BPU_TYPE_TENSOR_F32;
};

#endif  // _INPUT_FEATURE_ITERATOR_H_
