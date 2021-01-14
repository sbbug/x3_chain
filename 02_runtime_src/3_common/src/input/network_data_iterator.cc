// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/network_data_iterator.h"

#include <iostream>
#include <string>
#include <utility>

#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/tensor_utils.h"

#define RECV_QUEUE_SIZE 2
#define RECV_BUF_SIZE (1 * 1024 * 1024)

NetworkReceiver::NetworkReceiver()
    : socket_recv_(nullptr), zmq_context_(nullptr), zmq_msg_(0) {}

NetworkReceiver::~NetworkReceiver() { Fini(); }

bool NetworkReceiver::Init(const char *end_point) {
  zmq_context_ = zmq_ctx_new();
  socket_recv_ = zmq_socket(zmq_context_, ZMQ_PULL);

  int hwm = RECV_QUEUE_SIZE;
  int rc = zmq_setsockopt(socket_recv_, ZMQ_RCVHWM, &hwm, sizeof(int));
  if (rc != 0) {
    LOG(WARNING) << strerror(errno);
    return false;
  }

  int recv_buf_size = RECV_BUF_SIZE;
  rc = zmq_setsockopt(socket_recv_, ZMQ_RCVBUF, &recv_buf_size, sizeof(int));
  if (rc != 0) {
    LOG(WARNING) << strerror(errno);
    return false;
  }

  rc = zmq_bind(socket_recv_, end_point);
  if (rc != 0) {
    LOG(WARNING) << strerror(errno);
    return false;
  }

  LOG(INFO) << "Start to receive data from addr: " << end_point;
  return true;
}

void NetworkReceiver::SetDataType(hb_BPU_DATA_TYPE_E data_type) {
  data_type_ = data_type;
}

void NetworkReceiver::Fini() {
  zmq_close(socket_recv_);
  zmq_ctx_destroy(zmq_context_);
}

int NetworkReceiver::RecvImage(ImageTensor &image_tensor) {
  while (buf_queue_.empty()) {
    if (this->Recv() != OK) break;
  }

  while (buf_queue_.size() < 1) {
    if (this->Recv() != OK) break;
  }

  if (buf_queue_.size() < 1) {
    return TIMEOUT;
  }

  zmq_msg_t *msg = buf_queue_.front();
  buf_queue_.pop_front();
  int msg_size = zmq_msg_size(msg);
  char *msg_data = reinterpret_cast<char *>(zmq_msg_data(msg));

  ZMQMessage::ZMQMsg zmq_msg;
  zmq_msg.ParseFromString(std::string(msg_data, msg_size));

  if (zmq_msg.msg_type() == ZMQMessage::ZMQMsg_MsgType_IMAGE_MSG) {
    const ZMQMessage::ImageMsg &image_msg = zmq_msg.img_msg();

    image_tensor.ori_image_width = image_msg.image_width();
    image_tensor.ori_image_height = image_msg.image_height();
    image_tensor.image_name = image_msg.image_name();
    // TODO(yingxiang.hong): remove is_pad_resize
    image_tensor.is_pad_resize = true;
    auto &tensor = image_tensor.tensor;
    prepare_image_tensor(image_msg.image_dst_height(),
                         image_msg.image_dst_width(),
                         data_type_,
                         &tensor);
    const std::string &data = image_msg.image_data();
    memcpy(tensor.data.virAddr, data.data(), data.length());
    flush_tensor(&tensor);
  }

  zmq_msg_close(msg);
  delete msg;

  if (zmq_msg.msg_type() == ZMQMessage::ZMQMsg_MsgType_FINISH_MSG) {
    return FINISHED;
  } else {
    return OK;
  }
}

int NetworkReceiver::Recv() {
  int timeout = 10000;
  zmq_setsockopt(socket_recv_, ZMQ_RCVTIMEO, &timeout, sizeof(int));

  if (!zmq_msg_) {
    zmq_msg_ = new zmq_msg_t;
    zmq_msg_init(zmq_msg_);
  }

  while (true) {
    int rc = zmq_msg_recv(zmq_msg_, socket_recv_, 0);
    if (rc == -1) {
      // Assumed that only timeout happens here
      // TODO(yingxiang.hong): should handle other errors
      return TIMEOUT;
    }
    //    if (!zmq_msg_more(zmq_msg_)) {
    //      break;
    //    }
    break;
  }
  buf_queue_.push_back(zmq_msg_);
  zmq_msg_ = NULL;
  return OK;
}

int NetworkDataIterator::Init(std::string config_file,
                              std::string config_string) {
  network_receiver_ = new NetworkReceiver();
  int ret_code = DataIterator::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }

  return 0;
}

bool NetworkDataIterator::Next(ImageTensor *image_tensor) {
  while (true) {
    int ret_code = network_receiver_->RecvImage(*image_tensor);
    if (ret_code == TIMEOUT) {
      LOG(WARNING) << "Receive image timeout";
      continue;
    }
    if (ret_code == FINISHED) {
      is_finish_ = true;
    } else {
      image_tensor->frame_id = NextFrameId();
    }
    return !(ret_code == FINISHED);
  }
}
bool NetworkDataIterator::Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor){

     return true;
}
void NetworkDataIterator::Release(ImageTensor *image_tensor) {
  release_tensor(&image_tensor->tensor);
}

int NetworkDataIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("endpoint")) {
    endpoint = document["endpoint"].GetString();
  }

  if (document.HasMember("data_type")) {
    data_type_ =
        static_cast<hb_BPU_DATA_TYPE_E>(document["data_type"].GetInt());
  }

  network_receiver_->SetDataType(data_type_);

  if (network_receiver_->Init(endpoint.c_str())) {
    return 0;
  }

  return -1;
}

NetworkDataIterator::~NetworkDataIterator() {
  if (network_receiver_) {
    network_receiver_->Fini();
    delete network_receiver_;
    network_receiver_ = nullptr;
  }
}

bool NetworkDataIterator::HasNext() { return !is_finish_; }
