// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _INPUT_NETWORK_ITERATOR_H_
#define _INPUT_NETWORK_ITERATOR_H_

#include <deque>
#include <string>
#include <vector>

#include "input/data_iterator.h"
#include "protocol/zmq_msg.pb.h"
#include "zmq.h"

enum ReceiverStatus { OK = 0, TIMEOUT = -1, FINISHED = -2 };
class NetworkReceiver {
 public:
  NetworkReceiver();

  ~NetworkReceiver();

  bool Init(const char *end_point);

  void SetDataType(hb_BPU_DATA_TYPE_E data_type);

  int RecvImage(ImageTensor &image_tensor);

  void Fini();

 private:
  int Recv();

 private:
  void *socket_recv_;
  void *zmq_context_;
  std::deque<zmq_msg_t *> buf_queue_;
  zmq_msg_t *zmq_msg_;
  hb_BPU_DATA_TYPE_E data_type_ = BPU_TYPE_IMG_YUV444;
};

class NetworkDataIterator : public DataIterator {
 public:
  NetworkDataIterator() : DataIterator("network_data_iterator") {}

  /**
   * Init Image Data iterator from file
   * @param[in] config_file: config file path
   *        the config file should be in the json format
   *        for example:
   *        {
   *            "endpoint":  "tcp://*:6680"
   *        }
   * @param[in] config_string: config string
   *        same as config_file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Next Image Data from network
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

  ~NetworkDataIterator();

 private:
  int LoadConfig(std::string &config_string);

 private:
  NetworkReceiver *network_receiver_;
  std::string endpoint = "tcp://*:6680";
  hb_BPU_DATA_TYPE_E data_type_ = BPU_TYPE_IMG_YUV444;
};

#endif  // _INPUT_NETWORK_ITERATOR_H_
