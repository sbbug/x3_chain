// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _UTILS_IMAGE_UTILS_H_
#define _UTILS_IMAGE_UTILS_H_

#include <string>

#include "base/perception_common.h"
#include "input/input_data.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.h"

/**
 * Convert BGR to NV12
 * @param[in] bgr
 * @param[out] img_nv12
 */
void bgr_to_nv12(cv::Mat &bgr, cv::Mat &img_nv12);

/**
 * Draw perception result to frame
 * @param[in] image_tensor: ImageTensor data
 * @param[in] perception: Perception result
 * @param[out] mat: (bgr or gray)
 * @return 0 if success
 */
int draw_perception(ImageTensor *image_tensor,
                    Perception *perception,
                    cv::Mat &mat);

/**
 * ImageTensor to opencv mat
 * @param[in] image_tensor: ImageTensor data
 * @param[out] mat (bgr or gray)
 * @return 0 if success
 */
int image_tensor_to_mat(ImageTensor *image_tensor, cv::Mat &mat);

#endif  // _UTILS_IMAGE_UTILS_H_
