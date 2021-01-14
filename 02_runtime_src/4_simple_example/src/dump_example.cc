// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "bpu_predict_extension.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

#define EMPTY ""  // empty string

DEFINE_string(model_file, EMPTY, "Model file path");
DEFINE_string(input_file, EMPTY, "Input file path, binary file");
DEFINE_string(conv_mapping_file, EMPTY, "conv mapping file path");
DEFINE_string(conv_dump_path, EMPTY, "output path of conv output file");
DEFINE_int32(input_data_type, 2, "input data type");
DEFINE_int32(input_width, 1024, "input data width");
DEFINE_int32(input_height, 1024, "input data height");
DEFINE_int32(is_multi_input, 0, "is has multi input");
DEFINE_string(img_info_file, EMPTY, "img_info_file");
DEFINE_int32(img_info_input_width, 1, "img info input data width");
DEFINE_int32(img_info_input_height, 3, "img info input data height");

int main(int argc, char **argv) {
  // Init logging
  google::InitGoogleLogging(argv[0]);
  FLAGS_logbuflevel = google::INFO;
  FLAGS_logtostderr = true;

  // Parsing command line arguments
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Set conv mapping path
  if (!FLAGS_conv_mapping_file.empty()) {
    HB_BPU_setGlobalConfig(BPU_GLOBAL_CONV_MAPPING_FILE,
                           FLAGS_conv_mapping_file.c_str());
  }

  if (!FLAGS_conv_dump_path.empty()) {
    HB_BPU_setGlobalConfig(BPU_GLOBAL_CONV_DUMP_PATH,
                           FLAGS_conv_dump_path.c_str());
  }

  // Load model
  BPU_MODEL_S bpu_model;
  int ret_code = load_model_from_file(FLAGS_model_file, &bpu_model);
  LOG_IF(FATAL, ret_code != 0) << "Load model failed";
  DLOG(INFO) << "Load model success";
  std::cout << "---------------------------\n";
  std::cout << "input type: " << bpu_model.inputs[0].data_type << std::endl;
  std::vector<BPU_TENSOR_S> input_tensors(bpu_model.input_num);
  std::vector<BPU_TENSOR_S> output_tensors(bpu_model.output_num);

  hb_BPU_DATA_TYPE_E img_data_type =
      static_cast<hb_BPU_DATA_TYPE_E>(FLAGS_input_data_type);
  // Prepare input & output tensors
  prepare_image_tensor(
      FLAGS_input_height, FLAGS_input_width, img_data_type, &input_tensors[0]);

  // Is multi input model
  if (FLAGS_is_multi_input == 1) {
    prepare_image_tensor(FLAGS_img_info_input_height,
                         FLAGS_img_info_input_width,
                         BPU_TYPE_TENSOR_F32,
                         &input_tensors[1]);
  }

  ret_code =
      read_binary_file(FLAGS_input_file,
                       &input_tensors[0],
                       reinterpret_cast<char *>(input_tensors[0].data.virAddr));
  LOG_IF(FATAL, ret_code != 0)
      << "Prepare input tensor 0 failed," << HB_BPU_getErrorName(ret_code);
  flush_tensor(&input_tensors[0]);

  if (FLAGS_is_multi_input == 1) {
    ret_code = read_binary_file(
        FLAGS_img_info_file,
        reinterpret_cast<char *>(input_tensors[1].data.virAddr));
    LOG_IF(FATAL, ret_code != 0)
        << "Prepare input tensor 1 failed," << HB_BPU_getErrorName(ret_code);
    flush_tensor(&input_tensors[1]);
  }

  prepare_output_tensor(output_tensors, &bpu_model);

  // Run inference
  BPU_RUN_CTRL_S run_ctrl_s{1};
  BPU_TASK_HANDLE task_handle{};
  ret_code = HB_BPU_runModel(&bpu_model,
                             input_tensors.data(),
                             bpu_model.input_num,
                             output_tensors.data(),
                             bpu_model.output_num,
                             &run_ctrl_s,
                             true,
                             &task_handle);
  LOG_IF(FATAL, ret_code != 0)
      << "HB_BPU_runModel failed, " << HB_BPU_getErrorName(ret_code);

  // Release input & output tensors
  for (int i = 0; i < input_tensors.size(); i++) {
    release_tensor(&input_tensors[i]);
  }
  release_tensor(&input_tensors[1]);
  release_output_tensor(output_tensors);

  // Release model
  ret_code = HB_BPU_releaseModel(&bpu_model);
  LOG_IF(FATAL, ret_code != 0)
      << "Release model failed, " << HB_BPU_getErrorName(ret_code);

  LOG(INFO) << "Infer finished.";
}
