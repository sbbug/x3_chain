// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "input/camera_data_iterator.h"

#include "glog/logging.h"
#include "rapidjson/document.h"

int CameraDataIterator::Init(std::string config_file,
                             std::string config_string) {
  DLOG(INFO) << "Init camera data iterator";
  int ret_code = DataIterator::Init(config_file, config_string);
  if (ret_code != 0) {
    return -1;
  }
  return BPU_createCameraHandle(
      vio_cfg_.c_str(), cam_cfg_.c_str(), cam_index_, cam_port_, &cam_handle_);
}

bool CameraDataIterator::Next(ImageTensor *visible_image_tensor,ImageTensor *lwir_image_tensor){

     return true;
}

bool CameraDataIterator::Next(ImageTensor *image_tensor) {
  BPUCameraBuffer camera_buffer = nullptr;
  int ret_code = BPU_getCameraImageData(cam_handle_, &camera_buffer);
  if (ret_code != 0) {
    LOG(ERROR) << "Get camera image data failed:"
               << HB_BPU_getErrorName(ret_code);
    return false;
  }

  BPU_CAMERA_IMAGE_INFO_S camera_image_info;
  BPU_convertCameraInfo(&camera_image_info, camera_buffer);
  auto &image_info = camera_image_info.src_img;
  int y_len = image_info.height * image_info.width;
  int uv_len = y_len >> 1;
  BPU_DATA_SHAPE_S shape;
  BPU_TENSOR_S &tensor = image_tensor->tensor;

  tensor.data_type = BPU_TYPE_IMG_NV12_SEPARATE;
  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(tensor.data_type, nullptr, &h_idx, &w_idx, &c_idx);
  shape.ndim = 2;
  shape.d[0] = 1;
  shape.d[h_idx] = image_info.height;
  shape.d[w_idx] = image_info.width;
  shape.d[c_idx] = 3;

  tensor.data_shape = shape;
  tensor.aligned_shape = shape;
  tensor.aligned_shape.d[w_idx] = image_info.step;
  tensor.data.phyAddr = image_info.y_paddr;
  tensor.data.virAddr = reinterpret_cast<void *>(image_info.y_vaddr);
  tensor.data.memSize = y_len;
  tensor.data_ext.phyAddr = image_info.c_paddr;
  tensor.data_ext.virAddr = reinterpret_cast<void *>(image_info.c_vaddr);
  tensor.data_ext.memSize = uv_len;

  image_tensor->cam_id = camera_image_info.cam_id;
  image_tensor->frame_id = NextFrameId();
  image_tensor->timestamp = camera_image_info.timestamp;
  image_tensor->ori_image_height = image_info.height;
  image_tensor->ori_image_width = image_info.width;
  image_tensor->addition_info = camera_buffer;

  count_++;

  return true;
}

void CameraDataIterator::Release(ImageTensor *image_tensor) {
  auto camera_buffer =
      reinterpret_cast<BPUCameraBuffer>(image_tensor->addition_info);
  BPU_releaseCameraBuffer(cam_handle_, camera_buffer);
}

bool CameraDataIterator::HasNext() {
  return (frame_count_ == -1 || count_ < frame_count_);
}

int CameraDataIterator::LoadConfig(std::string &config_string) {
  rapidjson::Document document;
  document.Parse(config_string.data());

  if (document.HasParseError()) {
    LOG(ERROR) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("cam_cfg")) {
    cam_cfg_ = document["cam_cfg"].GetString();
  }

  if (document.HasMember("vio_cfg")) {
    vio_cfg_ = document["vio_cfg"].GetString();
  }

  if (document.HasMember("cam_index")) {
    cam_index_ = document["cam_index"].GetInt();
  }

  if (document.HasMember("cam_port")) {
    cam_port_ = document["cam_port"].GetInt();
  }

  if (document.HasMember("frame_count")) {
    frame_count_ = document["frame_count"].GetInt();
  }

  return 0;
}
