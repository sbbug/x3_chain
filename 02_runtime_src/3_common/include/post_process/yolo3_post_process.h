// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_YOLO3_POST_PROCESS_H_
#define _POST_PROCESS_YOLO3_POST_PROCESS_H_

#include <string>
#include <utility>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "post_process.h"

/**
 * Config definition for Yolo3
 */
struct Yolo3Config {
  std::vector<int> strides;
  std::vector<std::vector<std::pair<double, double>>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;
};

/**
 * Default yolo config
 * strides: [32, 16, 8]
 * anchors_table: [[[3.625, 2.8125], [ 4.875, 6.1875], [11.65625, 10.1875]],
 *                  [[1.875, 3.8125], [3.875, 2.8125], [ 3.6875,  7.4375]],
 *                  [[1.25,  1.625], [2.0, 3.75], [4.125, 2.875]]]
 * class_num: 80
 * class_names: ["person",      "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"]
 */
extern Yolo3Config default_yolo3_config;

class Yolo3PostProcessModule : public PostProcessModule {
 public:
  explicit Yolo3PostProcessModule(std::string instance_name)
      : PostProcessModule("yolo3_post_process", instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "score_threshold": 0.3,
   *        "nms_threshold": 0.45,
   *        "nms_top_k": 500,
   *        "yolov2": {
   *            "strides": ...
   *            "anchors_table": ...
   *            "class_num": ...
   *        }
   *    }
   * @param[in] config_string: config string
   * @return 0 if success
   */
  int Init(std::string config_file, std::string config_string);

  /**
   * Post process
   * @param[in] tensor: Model output tensors
   * @param[in] image_tensor: Input image tensor
   * @param[out] perception: Perception output data
   * @return 0 if success
   */
  int PostProcess(BPU_TENSOR_S *tensor,
                  ImageTensor *image_tensor,
                  Perception *perception);

 private:
  int LoadConfig(std::string &config_string);

  void PostProcess(BPU_TENSOR_S *tensor,
                   ImageTensor *frame,
                   int layer,
                   std::vector<Detection> &dets);

 private:
  Yolo3Config yolo3_config_ = default_yolo3_config;
  float score_threshold_ = 0.3;
  float nms_threshold_ = 0.45;
  int nms_top_k_ = 500;
};

#endif  // _POST_PROCESS_YOLO3_POST_PROCESS_H_
