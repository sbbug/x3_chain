// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include <chrono>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "bpu_predict_extension.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

#define EMPTY ""  // empty string

DEFINE_string(model_file, EMPTY, "Model file path");
DEFINE_string(preempted_model_file, EMPTY, "Preempted model file path");
DEFINE_string(input_file,
              EMPTY,
              "Input file path, original image file or binary file");

// core_num, default is 1
DEFINE_int32(core_num, 1, "core num");
DEFINE_int32(log_level,
             google::INFO,
             "Logging level (INFO=0, WARNING=1, ERROR=2, FATAL=3)");

int infer_once(BPU_MODEL_S &bpu_model,
               BPU_RUN_CTRL_S &run_ctrl_s,
               bool is_preempted,
               std::string input_file);

int main(int argc, char **argv) {
  // Parsing command line arguments
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << gflags::GetArgv() << std::endl;

  // Init logging
  google::InitGoogleLogging("");
  google::SetStderrLogging(0);
  FLAGS_colorlogtostderr = true;
  FLAGS_minloglevel = FLAGS_log_level;

  FLAGS_max_log_size = 200;
  FLAGS_logbufsecs = 0;
  FLAGS_logtostderr = true;

  // Load model
  BPU_MODEL_S bpu_model{};
  BPU_MODEL_S preempted_bpu_model{};
  int ret_code = load_model_from_file(FLAGS_model_file, &bpu_model);
  LOG_IF(FATAL, ret_code != 0)
      << "Load model failed, " << HB_BPU_getErrorName(ret_code);
  ret_code =
      load_model_from_file(FLAGS_preempted_model_file, &preempted_bpu_model);
  LOG_IF(FATAL, ret_code != 0)
      << "Load model failed, " << HB_BPU_getErrorName(ret_code);

  LOG(INFO) << "Model info:" << model_info(&bpu_model);
  LOG(INFO) << "Preempted model info:" << model_info(&preempted_bpu_model);

  BPU_RUN_CTRL_S run_ctrl_s{FLAGS_core_num};
  BPU_RUN_CTRL_S preempted_run_ctrl_s{FLAGS_core_num};

  std::thread t(infer_once,
                std::ref(bpu_model),
                std::ref(run_ctrl_s),
                false,
                FLAGS_input_file);
  std::thread preempted_t(infer_once,
                          std::ref(preempted_bpu_model),
                          std::ref(preempted_run_ctrl_s),
                          true,
                          FLAGS_input_file);
  t.join();
  preempted_t.join();

  // Release model
  ret_code = HB_BPU_releaseModel(&bpu_model);
  LOG_IF(FATAL, ret_code != 0)
      << "Release model failed, " << HB_BPU_getErrorName(ret_code);

  ret_code = HB_BPU_releaseModel(&preempted_bpu_model);
  LOG_IF(FATAL, ret_code != 0)
      << "Release model failed, " << HB_BPU_getErrorName(ret_code);

  LOG(INFO) << "Infer finished.";
}

int infer_once(BPU_MODEL_S &bpu_model,
               BPU_RUN_CTRL_S &run_ctrl_s,
               bool is_preempted,
               std::string input_file) {
  std::string msg = "normal model run: ";
  if (is_preempted) {
    msg = "preempted model run: ";
  } else {
    HB_BPU_setModelPrior(&bpu_model);
  }

  auto data_type = bpu_model.inputs[0].data_type;
  // Get input input_height and input_width
  BPU_LAYOUT_E layout = bpu_model.inputs[0].shape.layout;
  int input_height = 0;
  int input_width = 0;
  if (0 !=
      HB_BPU_getHW(
          data_type, &bpu_model.inputs[0].shape, &input_height, &input_width)) {
    std::cout << "HB_BPU_getHW failed!" << std::endl;
    exit(-1);
  }

  // Read original rgb image or y image
  cv::Mat img_mat;
  img_mat = cv::imread(input_file, cv::IMREAD_COLOR);
  // Resize to fit model input shape (input width & input height)
  cv::Mat resized_mat(input_height, input_width, img_mat.type());
  cv::resize(img_mat, resized_mat, resized_mat.size(), 0, 0);
  cv::Mat mat;

  BPU_TENSOR_S tensor;
  prepare_image_tensor(input_height, input_width, data_type, &tensor);
  int ori_img_width, ori_img_height;
  read_image_tensor(input_file, ori_img_width, ori_img_height, &tensor);
  // Prepare input tensors
  std::vector<BPU_TENSOR_S> input;
  input.push_back(tensor);

  // Prepare output buffer tensors
  std::vector<BPU_TENSOR_S> output;
  prepare_output_tensor(output, &bpu_model);

  // Run inference
  BPU_TASK_HANDLE task_handle{};
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::steady_clock::now();
    auto ret_code = HB_BPU_runModel(&bpu_model,
                                    input.data(),
                                    bpu_model.input_num,
                                    output.data(),
                                    bpu_model.output_num,
                                    &run_ctrl_s,
                                    true,
                                    &task_handle);
    LOG_IF(FATAL, ret_code != 0)
        << "Run model failed, " << HB_BPU_getErrorName(ret_code);
    auto end = std::chrono::steady_clock::now();
    std::cout << msg
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms" << std::endl;
  }

  // Release input & output buffer
  release_tensor(&input[0]);
  release_output_tensor(output);

  return 0;
}
