// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_TENSOR_UTILS_H_
#define _UTILS_TENSOR_UTILS_H_

#include <string>

#include "input/input_data.h"

/**
 * Align by 16
 */
#include <vector>

#include "bpu_predict_extension.h"
#define ALIGN_16(v) ((v + (16 - 1)) / 16 * 16)

/**
 * Prepare image tensor
 * @param[in] height
 * @param[in] width
 * @param[in] data_type: tensor data type
 * @param[out] tensor
 */
void prepare_image_tensor(int height,
                          int width,
                          hb_BPU_DATA_TYPE_E data_type,
                          BPU_TENSOR_S *tensor);

/**
 * Prepare feature tensor
 * @param[in] dims
 * @param[in] data_type
 * @param[out] tensor
 */
void prepare_feature_tensor(std::vector<int> &dims,
                            hb_BPU_DATA_TYPE_E data_type,
                            BPU_TENSOR_S *tensor);

/**
 *
 * @param[in] path:
 * @param[out] ori_width:
 * @param[out] ori_height:
 * @param[out] tensor:
 * @return 0 if success
 */
int read_image_tensor(std::string &path,
                      int &ori_width,
                      int &ori_height,
                      BPU_TENSOR_S *tensor);

/**
 * Flush tensor
 * @param[in] tensor: Tensor to be flushed
 */
void flush_tensor(BPU_TENSOR_S *tensor);

/**
 * Free tensor
 * @param tensor: Tensor to be released
 */
void release_tensor(BPU_TENSOR_S *tensor);

/**
 * Prepare output tensor
 * @param output
 * @param model
 */
void prepare_output_tensor(std::vector<BPU_TENSOR_S> &output,
                           BPU_MODEL_S *model);

/**
 * Release output tensor
 * @param output
 */
void release_output_tensor(std::vector<BPU_TENSOR_S> &output);
#endif  // _UTILS_TENSOR_UTILS_H_
