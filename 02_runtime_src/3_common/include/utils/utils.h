// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_UTILS_H_
#define _UTILS_UTILS_H_

#include <string>
#include <vector>

#include "bpu_predict_extension.h"
/**
 * Read model file
 * @param[in] file_path: file path
 * @param[out] bin: file binary content
 * @param[out] length: bin length
 * @return 0 if success otherwise -1
 */
int read_binary_file(std::string &file_path, char **bin, int *length);

/**
 * Read model file
 * @param[in] file_path: file path
 * @param[out] bin: file binary content
 * @return 0 if success otherwise -1
 */
int read_binary_file(std::string &file_path, char *bin);

/**
 * Read model file
 * @param[in] file_path: file path
 * @param[in] input_tensor: prepared input tensor
 * @param[out] bin: file binary content
 * @return 0 if success otherwise -1
 */
int read_binary_file(std::string &file_path,
                     BPU_TENSOR_S *input_tensor,
                     char *bin);

/**
 * Load model from file
 * @param[in] model_file: Model file path
 * @param[out] bpu_model
 * @return 0 if success otherwise error code
 */
int load_model_from_file(std::string &model_file, BPU_MODEL_S *bpu_model);

/**
 * Parse data_type enum to string
 * @param[in] data_type enum
 * @return data_type string
 */
std::string data_type_enum_to_string(BPU_DATA_TYPE_E data_type);

/**
 * Layout string name
 * @param[in] layout
 * @return layout string
 */
std::string layout_type_enum_to_string(BPU_LAYOUT_E layout);

/**
 * Get filename
 * @param path: file path
 * @return filename
 */
std::string get_file_name(std::string &path);

/**
 * Split str by sep
 * @param[in] str: str to split
 * @param[in] sep:
 * @param[out] tokens:
 * @param[in] limit:
 */
void split(std::string &str,
           char sep,
           std::vector<std::string> &tokens,
           int limit = -1);

/**
 * Reverse split str by sep
 * @param[in] str: str to split
 * @param[in] sep:
 * @param[out] tokens:
 * @param[in] limit:
 */
void rsplit(std::string &str,
            char sep,
            std::vector<std::string> &tokens,
            int limit = -1);

/**
 * NHWC to NCHW
 * @param[out] out_data0: channel/planar 0 data
 * @param[out] out_data1: channel/planar 1 data
 * @param[out] out_data2: channel/planar 2 data
 * @param[in] in_data:
 * @param[in] height: data height
 * @param[in] width: data width
 */
void nhwc_to_nchw(uint8_t *out_data0,
                  uint8_t *out_data1,
                  uint8_t *out_data2,
                  uint8_t *in_data,
                  int height,
                  int width);
/**
 * NCHW to NHWC
 * @param[out] out_data:
 * @param[in] in_data0: channel/planar 0 data
 * @param[in] in_data1: channel/planar 1 data
 * @param[in] in_data2: channel/planar 2 data
 * @param[in] height: data height
 * @param[in] width: data width
 */
void nchw_to_nhwc(uint8_t *out_data,
                  uint8_t *in_data0,
                  uint8_t *in_data1,
                  uint8_t *in_data2,
                  int height,
                  int width);

/**
 * Test whether it's the same shape
 * @param[in] lhs:
 * @param[in] rhs:
 * @return true if the shape is equal
 */
bool operator==(BPU_DATA_SHAPE_S &lhs, BPU_DATA_SHAPE_S &rhs);

/**
 * Squeeze shape (remove dims which size is 1)
 * @param[in] shape
 * @return squeezed shape
 */
BPU_DATA_SHAPE_S squeeze(BPU_DATA_SHAPE_S &shape);

/**
 *  Model info
 * @param[in] bpu_model: Bpu model
 */
std::string model_info(BPU_MODEL_S *bpu_model);

std::vector<std::string> s_split(const std::string& in, const std::string& delim);

#endif  // _UTILS_UTILS_H_
