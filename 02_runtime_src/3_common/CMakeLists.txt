cmake_minimum_required(VERSION 2.8)

include_directories(
        .
        include
        ${DEPS_ROOT}/bpu_predict/include
        ${DEPS_ROOT}/glog/include
        ${DEPS_ROOT}/gflags/include
        ${DEPS_ROOT}/libzmq/include
        ${DEPS_ROOT}/protobuf/include
        ${DEPS_ROOT}/opencv/include
        ${DEPS_ROOT}/libjpeg-turbo/include
        ${DEPS_ROOT}/rapidjson)

add_library(common
        protocol/zmq_msg.pb.cc
        protocol/meta.pb.cc
        protocol/meta_data.pb.cc
        protocol/common.pb.cc
        src/post_process/post_process.cc
        src/post_process/classification_post_process.cc
        src/post_process/fasterrcnn_post_process.cc
        src/post_process/rfcn_post_process.cc
        src/post_process/s3fd_post_process.cc
        src/post_process/segment_post_process.cc
        src/post_process/ssd_post_process.cc
        src/post_process/yolo2_post_process.cc
        src/post_process/yolo3_post_process.cc
        src/post_process/yolo5_post_process.cc
        src/post_process/yolo5_mutil_modal_post_process.cc
        src/input/data_iterator.cc
        src/input/image_list_data_iterator.cc
        src/input/mutil_modal_image_list_data_iterator.cc
        src/input/network_data_iterator.cc
        src/input/preprocessed_image_iterator.cc
        src/input/camera_data_iterator.cc
        src/input/feature_iteator.cc
        src/output/output.cc
        src/output/raw_output.cc
        src/output/image_list_output.cc
        src/output/video_output.cc
        src/output/client_output.cc
        src/utils/image_utils.cc
        src/utils/nms.cc
        src/utils/stop_watch.cc
        src/utils/tensor_utils.cc
        src/utils/utils.cc)
