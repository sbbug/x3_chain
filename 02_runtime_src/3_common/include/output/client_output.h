// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _OUTPUT_CLIENT_OUTPUT_H_
#define _OUTPUT_CLIENT_OUTPUT_H_

#include <string>

#include "base/perception_common.h"
#include "output.h"
#include "zmq.h"

class NetworkSender {
 public:
  NetworkSender();
  bool Init(const char *end_point);
  int Send(zmq_msg_t *msg, int flag);

  void Fini();
  ~NetworkSender();

 private:
  void *socket_snd_;
  void *zmq_context_;
};

class ClientOutputModule : public OutputModule {
 public:
  ClientOutputModule() : OutputModule("client_output") {}

  /**
   * Init ClientOutputModule
   * @param[in] config_file: config file
   *        config file should be in the json format
   *        for example:
   *        {
   *            "endpoint":  "tcp://*:5560"
   *        }
   * @param[in] config_string: config string
   *        same as config file
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Draw perception data
   * @param[in] image_tensor: Image tensor
   * @param[in] perception: perception data
   */
  void Write(ImageTensor *image_tensor, Perception *perception);

  ~ClientOutputModule();

 private:
  int LoadConfig(std::string &config_string);

  int Serialize(ImageTensor *image_tensor,
                Perception *perception,
                char **pb,
                int *pb_length);

 private:
  NetworkSender *network_sender_ = 0;
  std::string endpoint_ = "tcp://*:5560";
  int send_level_ = 0;
};

#endif  // _OUTPUT_CLIENT_OUTPUT_H_
