cmake_minimum_required(VERSION 2.8)

include_directories(
        ${CMAKE_SOURCE_DIR}/3_common/include
        ${DEPS_ROOT}/bpu_predict/include
        ${DEPS_ROOT}/glog/include
        ${DEPS_ROOT}/gflags/include
        ${DEPS_ROOT}/libzmq/include
        ${DEPS_ROOT}/protobuf/include
        ${DEPS_ROOT}/opencv/include)

link_directories(
        ${DEPS_ROOT}/bpu_predict/lib
        ${DEPS_ROOT}/system/lib
        ${DEPS_ROOT}/protobuf/lib
        ${DEPS_ROOT}/libzmq/lib
        ${DEPS_ROOT}/glog/lib
        ${DEPS_ROOT}/gflags/lib
        ${DEPS_ROOT}/opencv/lib)

if (${PLATFORM} STREQUAL "arm")
    SET(BPU_libs bpu_predict cnn_intf hbrt_bernoulli_aarch64 vio cam hbmedia isp iar isp_algo tinyalsa multimedia avformat
        avcodec avutil swresample gdcbin ion)
else ()
    SET(BPU_libs bpu_predict hbdk_sim_x86)
endif ()

SET(Link_libs
        common
        ${BPU_libs}
        protobuf
        gflags
        glog
        zmq
        zlib
        opencv_world
        dl
        pthread)

add_executable(example src/simple_example.cc)
add_executable(dump src/dump_example.cc)
add_executable(multi_input_example src/multi_input_example.cc)
add_executable(preempt_example src/preempt_example.cc)

target_link_libraries(example ${Link_libs})
target_link_libraries(dump ${Link_libs})
target_link_libraries(multi_input_example ${Link_libs})
target_link_libraries(preempt_example ${Link_libs})

install(TARGETS example dump multi_input_example preempt_example DESTINATION ${RELEASE_BIN_DIR}/)
