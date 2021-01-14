// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "output/client_output.h"

#include <cmath>
#include <thread>

#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "protocol/common.pb.h"
#include "protocol/meta.pb.h"
#include "protocol/meta_data.pb.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "utils/image_utils.h"

#define VERSION ((1 << 31) | (1 << 24))
#define MAX_DATA_SEND_CNT (5)

static void msg_free(void *data, void *hint) {
  delete[] reinterpret_cast<uint8_t *>(data);
}

static void set_image_info(CommonProto::Image *img_info,
                           ImageTensor *image_tensor,
                           int base,
                           Perception *perception) {
  if (perception->type == Perception::DET) {
    img_info->set_width(image_tensor->ori_width() / base);
    img_info->set_height(image_tensor->ori_height() / base);
    img_info->set_channel(3);
    img_info->set_time_stamp(image_tensor->timestamp);
    img_info->set_send_mode(0);
    img_info->set_format(CommonProto::NV12);
    img_info->set_index(0);
    img_info->set_count(1);
  } else if (perception->type == Perception::CLS) {
    img_info->set_width(image_tensor->width() / base);
    img_info->set_height(image_tensor->height() / base);
    img_info->set_channel(3);
    img_info->set_time_stamp(image_tensor->timestamp);
    img_info->set_send_mode(0);
    img_info->set_format(CommonProto::NV12);
    img_info->set_index(0);
    img_info->set_count(1);
  }
}

void cls_image_tensor_to_nv12(ImageTensor *image_tensor, cv::Mat &mat) {
  auto &tensor = image_tensor->tensor;
  auto data_type = tensor.data_type;
  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(
      data_type, &tensor.data_shape.layout, &h_idx, &w_idx, &c_idx);
  auto height = tensor.data_shape.d[h_idx];
  auto width = tensor.data_shape.d[w_idx];
  auto stride = tensor.aligned_shape.d[w_idx];

  if (data_type == BPU_TYPE_IMG_YUV444) {
    uint8_t *yuv = (uint8_t *)tensor.data.virAddr;
    mat = cv::Mat(height * 3 / 2, width, CV_8UC1);
    uint8_t *ynv12 = mat.ptr<uint8_t>();

    int y_size = height * width;
    uint8_t *nv12 = ynv12 + y_size;
    // copy yuv data
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        *ynv12++ = *(yuv + (i * width + j) * 3);
        if ((i % 2 == 0) && (j % 2 == 0)) {
          *nv12++ = *(yuv + (i * width + j) * 3 + 1);
          *nv12++ = *(yuv + (i * width + j) * 3 + 2);
        }
      }
    }
  }
}
static void set_camera_matrix(CommonProto::CameraMatrix *camera_matrix) {
  for (int i = 0; i < 9; ++i) {
    camera_matrix->mutable_mat_gnd2img()->Add(0);
    camera_matrix->mutable_mat_img2gnd()->Add(0);
    camera_matrix->mutable_mat_vcsgnd2img()->Add(0);
    camera_matrix->mutable_mat_img2vcsgnd()->Add(0);
  }
}

static void set_camera_param(CommonProto::CameraParam *camera_param) {
  camera_param->set_focal_u(1300.0);
  camera_param->set_focal_v(1300.0);
  camera_param->set_center_u(633.0);
  camera_param->set_center_v(382.0);
  camera_param->set_camera_x(0.016719);
  camera_param->set_camera_y(-1.361532);
  camera_param->set_camera_z(-1.442833);
  camera_param->set_pitch(0.030129);
  camera_param->set_yaw(0.008177);
  camera_param->set_roll(-0.000186);
  camera_param->set_fov(59.4);
  camera_param->set_type(0);

  CommonProto::DistortParam *distort = camera_param->mutable_distort();
  distort->mutable_param()->Clear();
  distort->add_param(0);
  distort->add_param(1);
  distort->add_param(2);
  distort->add_param(3);

  CommonProto::VCSParam *vcs = camera_param->mutable_vcs();
  vcs->mutable_rotation()->Clear();
  vcs->mutable_translation()->Clear();

  for (int i = 0; i < 3; ++i) {
    vcs->add_rotation(0);
    vcs->add_translation(0);
  }

  // Set camera matrix
  set_camera_matrix(camera_param->mutable_mat());
}

NetworkSender::NetworkSender() : socket_snd_(nullptr), zmq_context_(nullptr) {}

NetworkSender::~NetworkSender() { Fini(); }

bool NetworkSender::Init(const char *end_point) {
  zmq_context_ = zmq_ctx_new();
  socket_snd_ = zmq_socket(zmq_context_, ZMQ_PUB);

  int hwm = 1;
  zmq_setsockopt(socket_snd_, ZMQ_SNDHWM, &hwm, sizeof(int));
  int send_buf_size = MAX_DATA_SEND_CNT * 1024 * 1024;
  zmq_setsockopt(socket_snd_, ZMQ_SNDBUF, &send_buf_size, sizeof(int));
  int linger = 2000;
  int timeout = 2000;
  zmq_setsockopt(socket_snd_, ZMQ_LINGER, &linger, sizeof(int));
  zmq_setsockopt(socket_snd_, ZMQ_SNDTIMEO, &timeout, sizeof(int));

  int rc = zmq_bind(socket_snd_, end_point);
  if (rc != 0) {
    LOG(ERROR) << strerror(errno);
    return false;
  }

  return true;
}

int NetworkSender::Send(zmq_msg_t *msg, int flag) {
  bool retry = true;
  while (retry) {
    int rc = zmq_msg_send(msg, socket_snd_, flag);
    if (rc >= 0) {
      retry = false;
    } else {
      if (errno == EAGAIN || errno == EINTR) {
        DLOG(WARNING) << "Retry...";
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      } else {
        LOG(ERROR) << "ZMQ Get Error " << strerror(errno);
        return -1;
      }
    }
  }
  return 0;
}

void NetworkSender::Fini() {
  zmq_close(socket_snd_);
  zmq_ctx_destroy(zmq_context_);
}

int ClientOutputModule::Init(std::string config_file,
                             std::string config_string) {
  int ret_code = OutputModule::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return 0;
}

int ClientOutputModule::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("endpoint")) {
    endpoint_ = document["endpoint"].GetString();
  }

  network_sender_ = new NetworkSender;
  if (network_sender_->Init(endpoint_.c_str())) {
    return 0;
  }

  return -1;
}

ClientOutputModule::~ClientOutputModule() {
  if (network_sender_) {
    network_sender_->Fini();
    delete network_sender_;
    network_sender_ = nullptr;
  }
}

void ClientOutputModule::Write(ImageTensor *image_tensor,
                               Perception *perception) {
  if (perception->type == Perception::DET) {
    auto height = image_tensor->ori_image_height;
    auto width = image_tensor->ori_image_width;
    if (height % 2 || width % 2) {
      LOG(INFO) << "client send error! the original image is unqualified, "
                   "can't convert to nv12!";
      return;
    }

    {
      // Send meta data
      char *pb;
      int pb_len;
      this->Serialize(image_tensor, perception, &pb, &pb_len);
      zmq_msg_t msg_meta;
      zmq_msg_init_data(&msg_meta, pb, pb_len, msg_free, nullptr);
      network_sender_->Send(&msg_meta, ZMQ_NOBLOCK | ZMQ_SNDMORE);
      zmq_msg_close(&msg_meta);
    }

    {
      cv::Mat mat;
      image_tensor_to_mat(image_tensor, mat);
      cv::Mat ori_nv12;
      bgr_to_nv12(mat, ori_nv12);
      auto y_len = height * width;
      auto uv_len = height * width / 2;
      auto img_len = y_len + uv_len;
      uint8_t *img_data = new uint8_t[img_len];
      uint8_t *nv12_data = ori_nv12.data;
      memcpy(img_data, nv12_data, y_len + uv_len);
      zmq_msg_t msg_img;
      zmq_msg_init_data(&msg_img, img_data, img_len, msg_free, nullptr);
      network_sender_->Send(&msg_img, ZMQ_NOBLOCK);
      zmq_msg_close(&msg_img);
      LOG(INFO) << "client send finished";
    }
  } else if (perception->type == Perception::CLS) {
    auto height = image_tensor->height();
    auto width = image_tensor->width();
    if (height % 2 || width % 2) {
      LOG(INFO) << "client send error! the original image is unqualified, "
                   "can't convert to nv12!";
      return;
    }
    {
      // Send meta data
      char *pb;
      int pb_len;
      this->Serialize(image_tensor, perception, &pb, &pb_len);
      zmq_msg_t msg_meta;
      zmq_msg_init_data(&msg_meta, pb, pb_len, msg_free, nullptr);
      network_sender_->Send(&msg_meta, ZMQ_NOBLOCK | ZMQ_SNDMORE);
      zmq_msg_close(&msg_meta);
    }

    {
      cv::Mat mat;
      cls_image_tensor_to_nv12(image_tensor, mat);
      auto y_len = height * width;
      auto uv_len = height * width / 2;
      auto img_len = y_len + uv_len;
      uint8_t *img_data = new uint8_t[img_len];
      uint8_t *nv12_data = mat.data;
      memcpy(img_data, nv12_data, y_len + uv_len);
      zmq_msg_t msg_img;
      zmq_msg_init_data(&msg_img, img_data, img_len, msg_free, nullptr);
      network_sender_->Send(&msg_img, ZMQ_NOBLOCK);
      zmq_msg_close(&msg_img);
      LOG(INFO) << "client send finished";
    }
  }
}

int ClientOutputModule::Serialize(ImageTensor *image_tensor,
                                  Perception *perception,
                                  char **pb,
                                  int *pb_length) {
  int base = std::pow(2, send_level_ / 4);
  Meta::Meta serial_meta;
  serial_meta.set_proto_version(1);
  serial_meta.set_version(VERSION);
  serial_meta.set_frame_id(image_tensor->frame_id);
  auto img_frame = serial_meta.mutable_img_frame();
  set_image_info(img_frame, image_tensor, base, perception);

  MetaData::Data *meta_data = serial_meta.mutable_data();
  meta_data->set_version(VERSION);
  meta_data->set_frame_id(image_tensor->frame_id);

  // Set image_tensor
  CommonProto::Image *img_info = meta_data->add_image();
  set_image_info(img_info, image_tensor, base, perception);

  // Set data descriptor
  MetaData::DataDescriptor *data_desc = meta_data->add_data_descriptor();
  data_desc->set_type("image");
  MetaData::SerializedData *ser_data = data_desc->mutable_data();
  ser_data->set_type("image");
  ser_data->set_channel(0);
  static std::vector<uint8_t> meta;
  meta.resize(img_info->ByteSize());
  img_info->SerializeToArray(meta.data(), meta.size());
  ser_data->set_proto(meta.data(), meta.size());
  ser_data->set_with_data_field(true);

  // Set camera param & matrix
  CommonProto::CameraParam *camera_param = meta_data->add_camera();
  set_camera_param(camera_param);
  set_camera_param(meta_data->add_camera_default());
  set_camera_matrix(meta_data->add_camera_matrix());

  // Set perception data
  MetaData::StructurePerception *percep =
      meta_data->mutable_structure_perception();
  // Set ObstacleRaws
  CommonProto::ObstacleRaws *obs_raws = percep->add_obstacles_raws();
  obs_raws->set_cam_id(image_tensor->cam_id);

  if (perception->type == Perception::DET) {
    auto &dets = perception->det;
    for (int i = 0; i < dets.size(); i++) {
      CommonProto::ObstacleRaw *obr = obs_raws->add_obstacle();
      CommonProto::Rect *rect = obr->mutable_rect();
      auto &det = dets[i];
      auto &box = det.bbox;
      rect->set_left(box.xmin / base);
      rect->set_top(box.ymin / base);
      rect->set_right(box.xmax / base);
      rect->set_bottom(box.ymax / base);
      obr->set_conf(det.score);
      if (det.id < 20) {
        obr->set_model(det.id);
      } else {
        obr->set_model(19);
      }
      obr->set_source(1);
      if (det.class_name) {
        obr->add_property(det.id);
        obr->add_property_name(det.class_name);
        obr->add_property_type(det.id);
      }
    }
  } else if (perception->type == Perception::CLS) {
    auto &cls = perception->cls[0];
    CommonProto::ObstacleRaw *obr = obs_raws->add_obstacle();
    CommonProto::Rect *rect = obr->mutable_rect();
    rect->set_left(0);
    rect->set_top(0);
    rect->set_right(100 / base);
    rect->set_bottom(100 / base);
    obr->set_conf(cls.score);
    obr->set_model(0);
    obr->set_source(1);

    if (cls.class_name) {
      obr->add_property(cls.id);
      obr->add_property_name(cls.class_name);
      obr->add_property_type(cls.id);
    }
  }

  *pb_length = serial_meta.ByteSize();
  *pb = new char[*pb_length];
  serial_meta.SerializeToArray(*pb, *pb_length);

  return 0;
}
