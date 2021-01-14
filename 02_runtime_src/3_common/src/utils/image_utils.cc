// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "utils/image_utils.h"

#include <fstream>
#include <string>
#include<iomanip>
#include "glog/logging.h"

void bgr_to_nv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);

  uint8_t *yuv = yuv_mat.ptr<uint8_t>();
  img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  uint8_t *ynv12 = img_nv12.ptr<uint8_t>();

  int uv_height = height / 2;
  int uv_width = width / 2;

  // copy y data
  int y_size = height * width;
  memcpy(ynv12, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
}

int draw_perception(ImageTensor *frame, Perception *perception, cv::Mat &mat) {
  if (image_tensor_to_mat(frame, mat) != 0) {
    return -1;
  }

  static cv::Scalar colors[] = {
      cv::Scalar(255, 0, 0),     // red
      cv::Scalar(255, 165, 0),   // orange
      cv::Scalar(255, 255, 0),   // yellow
      cv::Scalar(0, 255, 0),     // green
      cv::Scalar(0, 0, 255),     // blue
      cv::Scalar(75, 0, 130),    // indigo
      cv::Scalar(238, 130, 238)  // violet
  };

  if (perception->type == Perception::DET) {
    auto &det = perception->det;
    for (int i = 0; i < det.size(); i++) {
      auto &color = colors[det[i].id % 7];
      Bbox &bbox = det[i].bbox;
      cv::rectangle(mat,
                    cv::Point(bbox.xmin, bbox.ymin),
                    cv::Point(bbox.xmax, bbox.ymax),
                    color);
      std::stringstream text_ss;
      text_ss << det[i].id << " " << det[i].class_name << ":" << std::fixed
              << std::setprecision(2) << det[i].score;
      cv::putText(mat,
                  text_ss.str(),
                  cv::Point(bbox.xmin, std::abs(bbox.ymin - 5)),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.5,
                  color,
                  1,
                  cv::LINE_AA);
    }
  } else if (perception->type == Perception::SEG) {
    // TODO(horizon.ai):
  } else {
    auto &cls = perception->cls;
    for (int i = 0; i < cls.size(); i++) {
      auto &c = cls[i];
      auto &color = colors[c.id % 7];
      std::stringstream text_ss;
      text_ss << c.id << ":" << std::fixed << std::setprecision(6) << c.score;
      cv::putText(mat,
                  text_ss.str(),
                  cv::Point(5, 20 + 10 * i),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.5,
                  color,
                  1,
                  cv::LINE_AA);
    }
  }
  return 0;
}

int image_tensor_to_mat(ImageTensor *image_tensor, cv::Mat &mat) {
  auto &tensor = image_tensor->tensor;
  auto data_type = tensor.data_type;
  int h_idx, w_idx, c_idx;
  HB_BPU_getHWCIndex(
      data_type, &tensor.data_shape.layout, &h_idx, &w_idx, &c_idx);
  auto height = tensor.data_shape.d[h_idx];
  auto width = tensor.data_shape.d[w_idx];
  auto stride = tensor.aligned_shape.d[w_idx];

  if (data_type == BPU_TYPE_IMG_YUV444) {
    mat.create(
        image_tensor->ori_image_height, image_tensor->ori_image_width, CV_8UC3);
    cv::Mat resized(height, width, CV_8UC3);
    memcpy(resized.data, tensor.data.virAddr, height * width * 3);
    cv::Mat yuv;
    cv::resize(resized, yuv, mat.size(), 0, 0);
    cv::cvtColor(yuv, mat, CV_YUV2BGR);
  } else if (data_type == BPU_TYPE_IMG_Y) {
    mat.create(
        image_tensor->ori_image_height, image_tensor->ori_image_width, CV_8UC1);
    cv::Mat resized(height, width, CV_8UC1);

    uint8_t *data = resized.data;
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor.data.virAddr);

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        *data++ = data0[h * stride + w];
      }
    }
    cv::resize(resized, mat, mat.size(), 0, 0);
  } else if (data_type == BPU_TYPE_IMG_YUV_NV12) {
    mat.create(
        image_tensor->ori_image_height, image_tensor->ori_image_width, CV_8UC3);
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
    cv::Mat resized;

    uint8_t *data = nv12.data;
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor.data.virAddr);
    uint8_t *data1 = data0 + height * stride;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        *data++ = data0[h * stride + w];
      }
    }
    for (int i = 0; i < height * width / 2; i++) {
      *data++ = *data1++;
    }
    cv::cvtColor(nv12, resized, cv::COLOR_YUV2BGR_NV12);
    cv::resize(resized, mat, mat.size(), 0, 0);
  } else if (data_type == BPU_TYPE_IMG_NV12_SEPARATE) {
    mat.create(
        image_tensor->ori_image_height, image_tensor->ori_image_width, CV_8UC3);
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
    cv::Mat resized;

    uint8_t *data = nv12.data;
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor.data.virAddr);
    uint8_t *data1 = reinterpret_cast<uint8_t *>(tensor.data_ext.virAddr);
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        *data++ = data0[h * stride + w];
      }
    }
    for (int i = 0; i < height * width / 2; i++) {
      *data++ = *data1++;
    }
    cv::cvtColor(nv12, resized, cv::COLOR_YUV2BGR_NV12);
    cv::resize(resized, mat, mat.size(), 0, 0);
  } else if (data_type == BPU_TYPE_IMG_BGRP) {
    mat.create(
        image_tensor->ori_image_height, image_tensor->ori_image_width, CV_8UC3);
    cv::Mat resized(height, width, CV_8UC3);
    int offset = height * width;
    nchw_to_nhwc(resized.ptr<uint8_t>(),
                 reinterpret_cast<uint8_t *>(tensor.data.virAddr),
                 reinterpret_cast<uint8_t *>(tensor.data.virAddr) + offset,
                 reinterpret_cast<uint8_t *>(tensor.data.virAddr) + offset * 2,
                 height,
                 width);
    cv::resize(resized, mat, mat.size(), 0, 0);
  } else if (data_type == BPU_TYPE_IMG_RGBP) {
    mat.create(
        image_tensor->ori_image_height, image_tensor->ori_image_width, CV_8UC3);
    cv::Mat resized(height, width, CV_8UC3);
    int offset = height * width;
    nchw_to_nhwc(resized.ptr<uint8_t>(),
                 reinterpret_cast<uint8_t *>(tensor.data.virAddr),
                 reinterpret_cast<uint8_t *>(tensor.data.virAddr) + offset,
                 reinterpret_cast<uint8_t *>(tensor.data.virAddr) + offset * 2,
                 height,
                 width);
    cv::Mat bgr;
    cv::cvtColor(resized, bgr, CV_RGB2BGR);
    cv::resize(bgr, mat, mat.size(), 0, 0);
  } else {
    LOG(ERROR) << "Not implemented for " << data_type << " yet";
    return -1;
  }
  return 0;
}
