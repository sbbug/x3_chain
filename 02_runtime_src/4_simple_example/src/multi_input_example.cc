// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include <fstream>

#include "bpu_predict_extension.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "input/data_iterator.h"
#include "output/output.h"
#include "post_process/post_process.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

#define EMPTY ""

DEFINE_int32(log_level,
             google::INFO,
             "Logging level (INFO=0, WARNING=1, ERROR=2, FATAL=3)");
DEFINE_string(input_type,
              EMPTY,
              "Input type can be one of [network, image, binary, camera]");
DEFINE_string(input_config_string, EMPTY, "Json config for input module");
DEFINE_string(input_config_file, EMPTY, "Json config file for input module");
DEFINE_string(model_name, EMPTY, "Model name");
DEFINE_string(model_file, EMPTY, "Model file");
DEFINE_int32(core_num, 1, "core mode (1 for single core, 2 for dual core)");
DEFINE_string(post_process_config_string,
              EMPTY,
              "Json config for post process module");
DEFINE_string(post_process_config_file,
              EMPTY,
              "Json config file for post process module");
DEFINE_string(output_type,
              EMPTY,
              "Output type can be one of [raw, image, video, client]");
DEFINE_string(output_config_string,
              EMPTY,
              "Json string config for output module");
DEFINE_string(output_config_file, EMPTY, "Json config file for output module");

void prepare_img_info_tensor(BPU_TENSOR_S *input_tensor, float *img_info_data);

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
  BPU_MODEL_S bpu_model;
  int ret_code = load_model_from_file(FLAGS_model_file, &bpu_model);
  LOG_IF(FATAL, ret_code != 0) << "Load model failed";

  // Init input data source
  DataIterator *data_iterator = DataIterator::GetImpl(FLAGS_input_type);
  data_iterator->Init(FLAGS_input_config_file, FLAGS_input_config_string);

  // Init the output
  OutputModule *output = OutputModule::GetImpl(FLAGS_output_type);
  output->Init(FLAGS_output_config_file, FLAGS_output_config_string);

  // Input post process
  PostProcessModule *post_process_module =
      PostProcessModule::GetImpl(FLAGS_model_name);
  post_process_module->Init(FLAGS_post_process_config_file,
                            FLAGS_post_process_config_string);

  ImageTensor data;
  std::vector<BPU_TENSOR_S> input_tensors(bpu_model.input_num);
  std::vector<BPU_TENSOR_S> output_tensors(bpu_model.output_num);
  BPU_RUN_CTRL_S run_ctrl_s{FLAGS_core_num};
  BPU_TASK_HANDLE task_handle{};

  Stopwatch whole_watch;
  Stopwatch infer_watch;
  Stopwatch post_process_watch;

  // for faster rcnn
  std::vector<float> img_info_data{1024.f, 1024.f, 1.f};
  // Run loop
  while (data_iterator->HasNext()) {
    // Fetch one frame
    if (!data_iterator->Next(&data)) {
      continue;
    }

    // Prepare input & output tensors
    input_tensors[0] = data.tensor;

    BPU_TENSOR_S img_info_tensor;
    prepare_img_info_tensor(&img_info_tensor, img_info_data.data());
    input_tensors[1] = img_info_tensor;

    // Here we suppose that data type and shape match the model
    //  or you can do convert & resize here
    LOG_IF(ERROR, data.tensor.data_type != bpu_model.inputs[0].data_type)
        << "Input type does not match, you should convert it";

    prepare_output_tensor(output_tensors, &bpu_model);

    // Run inference
    whole_watch.Start();
    infer_watch.Start();
    ret_code = HB_BPU_runModel(&bpu_model,
                               input_tensors.data(),
                               bpu_model.input_num,
                               output_tensors.data(),
                               bpu_model.output_num,
                               &run_ctrl_s,
                               true,
                               &task_handle);
    LOG_IF(FATAL, ret_code != 0)
        << "Run model failed:" << HB_BPU_getErrorName(ret_code);
    infer_watch.Stop();

    // Post process
    post_process_watch.Start();
    Perception perception;
    post_process_module->PostProcess(output_tensors.data(), &data, &perception);
    post_process_watch.Stop();
    whole_watch.Stop();

    // Write output
    output->Write(&data, &perception);

    // Output tensors
    release_output_tensor(output_tensors);

    // Release frame data
    data_iterator->Release(&data);

    LOG(INFO) << "Image:" << data.image_name << ", infer result:" << perception;
  }

  std::stringstream ss;
  ss << "Whole process statistics:" << whole_watch
     << ", Infer stage statistics:" << infer_watch
     << ", Post process stage statistics:" << post_process_watch << std::endl;
  LOG(INFO) << ss.str();

  // Release input module
  delete data_iterator;

  // Release post process module
  delete post_process_module;

  // Release model
  HB_BPU_releaseModel(&bpu_model);
}

void prepare_img_info_tensor(BPU_TENSOR_S *input_tensor, float *img_info_data) {
  auto &mem = input_tensor->data;
  input_tensor->data_type = BPU_TYPE_TENSOR_F32;
  int image_length = 3 * 4;
  int ret = HB_SYS_bpuMemAlloc("mem_in", image_length, true, &mem);
  if (ret != 0) {
    return;
  }

  memcpy(mem.virAddr, img_info_data, image_length);
  ret = HB_SYS_flushMemCache(&mem, HB_SYS_MEM_CACHE_CLEAN);
  if (ret != 0) {
    return;
  }

  auto &shape = input_tensor->data_shape;
  input_tensor->data_type = BPU_TYPE_TENSOR_F32;
  input_tensor->data_shape.layout = BPU_LAYOUT_NHWC;
  input_tensor->aligned_shape = shape;
}
