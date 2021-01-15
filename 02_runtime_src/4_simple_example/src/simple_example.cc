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
              "Input type can be one of [network, image, preprocessed_image, "
              "feature, camera]");
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
DEFINE_bool(enable_post_process, true, "Is model need post process");

int main(int argc, char **argv) {
  // Parsing command line arguments
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "-------------------" << gflags::GetArgv() << std::endl;

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
  LOG(INFO) << "Model info:------------" << model_info(&bpu_model);
  std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
  // Init input data source
  DataIterator *data_iterator = DataIterator::GetImpl(FLAGS_input_type);
  data_iterator->Init(FLAGS_input_config_file, FLAGS_input_config_string);
  std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
  // Init the output
  OutputModule *output = OutputModule::GetImpl(FLAGS_output_type);
  output->Init(FLAGS_output_config_file, FLAGS_output_config_string);
  std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
  // Input post process
  PostProcessModule *post_process_module =
      PostProcessModule::GetImpl(FLAGS_model_name);
  post_process_module->Init(FLAGS_post_process_config_file,
                            FLAGS_post_process_config_string);

  ImageTensor visible_data;
  ImageTensor lwir_data;
  std::cout<<"-----------bpu_model.input_num"<<bpu_model.input_num<<std::endl;
  std::cout<<"-----------bpu_model.output_num"<<bpu_model.output_num<<std::endl;

  std::vector<BPU_TENSOR_S> input_tensors(bpu_model.input_num);
  std::vector<BPU_TENSOR_S> output_tensors(bpu_model.output_num);
  BPU_RUN_CTRL_S run_ctrl_s{FLAGS_core_num};
  BPU_TASK_HANDLE task_handle{};

  Stopwatch whole_watch;
  Stopwatch infer_watch;
  Stopwatch post_process_watch;

  // Run loop
  while (data_iterator->HasNext()) {
    // Fetch one frame
    if (!data_iterator->Next(&visible_data,&lwir_data)) {
      continue;
    }

    std::cout<<visible_data<<lwir_data<<std::endl;
    std::cout<<visible_data.image_name<<lwir_data.image_name<<std::endl;
    std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
    // Prepare input & output tensors
    input_tensors[0] = visible_data.tensor;
    input_tensors[1] = lwir_data.tensor;
    // Here we suppose that data type and shape match the model
    //  or you can do convert & resize here
    // LOG_IF(ERROR, data.tensor.data_type != bpu_model.inputs[0].data_type)
    //     << "Input type does not match, you should convert it";

    //    BPU_DATA_SHAPE_S squeezed_shape = squeeze(bpu_model.inputs[0].shape);
    //    LOG_IF(ERROR, !(data.tensor.data_shape == squeezed_shape))
    //        << "Input shape do not match, you should resize it";
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
    /*
    output_tensors:
     0:1x84x84x24
     1:1x42x42x24
     2:1x21x21x24
     3:1x84x84x24
     4:1x42x42x24
     5:1x21x21x24
    */
    for(int i=0;i<output_tensors.size();i++){
       std::cout<<output_tensors[i].data_shape.d[0]<<"X";
       std::cout<<output_tensors[i].data_shape.d[1]<<"X";
       std::cout<<output_tensors[i].data_shape.d[2]<<"X";
       std::cout<<output_tensors[i].data_shape.d[3]<<std::endl;
    }
    Perception perception;
    if (FLAGS_enable_post_process) {
      // Post process
      post_process_watch.Start();
      post_process_module->PostProcess(
          output_tensors.data(), &visible_data, &perception);
      post_process_watch.Stop();
      whole_watch.Stop();
      // Write output
      output->Write(&visible_data, &perception);
    }

    if (!FLAGS_enable_post_process) {
      whole_watch.Stop();
    }
    // Output tensors
    release_output_tensor(output_tensors);

    // Release frame data
    data_iterator->Release(&visible_data);
    data_iterator->Release(&lwir_data);

    if (FLAGS_enable_post_process) {
      LOG(INFO) << "Image:" << visible_data.image_name
                << ", infer result:" << perception << std::endl;
    } else {
      LOG(INFO) << "Image:" << visible_data.image_name << ", infer end.";
    }
  }

  std::stringstream ss;
  ss << "Whole process statistics:" << whole_watch
     << ", Infer stage statistics:" << infer_watch;

  if (FLAGS_enable_post_process) {
    ss << ", Post process stage statistics:" << post_process_watch << std::endl;
  } else {
    ss << std::endl;
  }
  LOG(INFO) << ss.str();

  // Release input module
  delete data_iterator;

  // Release post process module
  delete post_process_module;

  // Release model
  HB_BPU_releaseModel(&bpu_model);

}
