// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_IMAGE_LIST_ITERATOR_H_
#define _INPUT_IMAGE_LIST_ITERATOR_H_

#include <string>

#include "data_iterator.h"

class ImageListDataIterator : public DataIterator {
 public:
  ImageListDataIterator() : DataIterator("image_list_data_iterator") {}

  /**
   * Init Image Data iterator from file
   * @param[in] config_file: config file
   *        the file content should be in the json format
   *        for example:
   *        {
   *            "image_list_file" : "image_list.txt" #  one image file per line
   *            "width": 214,
   *            "height": 214,
   *            "data_type": 2
   *        }
   * @param[in] config_string: config string
   *        same as config_file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Release image_tensor
   * @param[in] image_tensor: image tensor to be released
   */
  void Release(ImageTensor *image_tensor);

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
   * Check if has next image
   * @return 0 if finish
   */
  bool HasNext();

  ~ImageListDataIterator();

 private:
  int LoadConfig(std::string &config_string);

 private:
  std::string image_list_file_;
  std::ifstream ifs_;
  int width_;
  int height_;
  hb_BPU_DATA_TYPE_E data_type_ = BPU_TYPE_IMG_YUV444;
};

#endif  // _INPUT_IMAGE_LIST_ITERATOR_H_
