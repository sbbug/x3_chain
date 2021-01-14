// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _POST_PROCESS_YOLO5_POST_PROCESS_H_
#define _POST_PROCESS_YOLO5_POST_PROCESS_H_

#include <string>
#include <utility>
#include <vector>

#include "base/perception_common.h"
#include "bpu_predict_extension.h"
#include "post_process.h"

/**
 * Config definition for Yolo5
 */
struct Yolo5Config {
  std::vector<int> strides;
  std::vector<std::vector<std::pair<double, double>>> anchors_table;
  int class_num;
  std::vector<std::string> class_names;
};

/**
 * Default yolo config
 * strides: [8, 16, 32]
 * anchors_table: [[[10, 13], [16,30], [33,23]],
 *                  [[30,61], [62,45], [59,119]],
 *                  [[116,90], [156,198], [373,326]]]
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
extern Yolo5Config default_yolo5_config;

class Yolo5PostProcessModule : public PostProcessModule {
 public:
  explicit Yolo5PostProcessModule(std::string instance_name)
      : PostProcessModule("yolo5_post_process", instance_name) {}

  /**
   * Load configuration from file
   * @param[in] config_file: config file path
   *    Config file should be json format
   *    for example:
   *    {
   *        "score_threshold": 0.3,
   *        "nms_threshold": 0.45,
   *        "nms_top_k": 500,
   *        "yolov5": {
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
  Yolo5Config yolo5_config_ = default_yolo5_config;
  float score_threshold_ = 0.001;
  float nms_threshold_ = 0.65;
  int nms_top_k_ = 5000;
};

#endif  // _POST_PROCESS_YOLO5_POST_PROCESS_H_
