# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import google.protobuf
import zmq_msg_pb2
import zmq
import time
import sys
import random
import numpy as np
import time
import ast
import os
import argparse
from data_loader import *


class ZmqSenderClient:
    def __init__(self, end_point):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.connect(end_point)

    def __del__(self):
        self.socket.close()
        self.context.destroy()

    def send_image_msg(self, img, image_name, image_type, org_h, org_w, dst_h,
                       dst_w):
        zmq_msg = zmq_msg_pb2.ZMQMsg()
        image_msg = zmq_msg.img_msg
        image_msg.image_data = img.tobytes()
        image_msg.image_width = org_w
        image_msg.image_height = org_h
        image_msg.image_name = image_name
        image_msg.image_format = image_type
        image_msg.image_dst_width = dst_w
        image_msg.image_dst_height = dst_h
        zmq_msg.msg_type = zmq_msg_pb2.ZMQMsg.IMAGE_MSG

        msg = zmq_msg.SerializeToString()
        while True:
            try:
                self.socket.send(msg, zmq.NOBLOCK)
                break
            except zmq.ZMQError:
                continue
        return 0

    def over(self):
        time.sleep(2)
        while True:
            try:
                zmq_msg = zmq_msg_pb2.ZMQMsg()
                zmq_msg.msg_type = zmq_msg_pb2.ZMQMsg.FINISH_MSG
                msg = zmq_msg.SerializeToString()
                self.socket.send(msg, zmq.NOBLOCK)
                break
            except zmq.ZMQError:
                continue


def send_images(ip, algo_name, input_file_path, is_input_preprocessed,
                image_count, image_type):
    end_point = "tcp://" + ip + ":6680"

    print("start to send data to %s ..." % end_point)

    zmq_sender = ZmqSenderClient(end_point)

    paths = os.listdir(input_file_path)  # 列出文件夹下所有的目录

    if image_count > len(paths):
        print('image count is too large')
        image_count = len(paths)

    # TODO random select images when image_count < len(paths)
    path_list = np.arange(image_count)
    current_count = 0
    for i in path_list:
        path = os.path.join(input_file_path, paths[i])
        if os.path.isdir(path):
            print("input path contains a dir !!!")
            return
        if not is_input_preprocessed:
            org_h, org_w, dst_h, dst_w, data = image_loader(
                path, algo_name, image_type)
            image_name = paths[i].encode('utf-8')
        else:
            res = str.split(paths[i], '_')
            data = np.fromfile(path, dtype=np.float32)
            # {name}_{org_h}_{org_w}_{dst_h}_{dst_w}.bin
            image_name = '_'.join(res[:-4]).encode('utf-8')
            org_h = int(res[-4])
            org_w = int(res[-3])
            dst_h = int(res[-2])
            dst_w = int(res[-1].split('.')[0])
        t = time.time()
        start_time = int(round(t * 1000))
        zmq_sender.send_image_msg(data, image_name, image_type, org_h, org_w,
                                  dst_h, dst_w)
        end_time = int(round(t * 1000))
        current_count = current_count + 1
        print(
            current_count, 'image name is: %s' % image_name.decode('utf-8'),
            ' send image data message done, take %d ms' %
            (end_time - start_time))
    zmq_sender.over()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algo-name',
        type=str,
        required=True,
        help='model type, such as mobilenetv1, mobilenetv2, '
        'yolov2, yolov3')

    parser.add_argument(
        '--input-file-path',
        type=str,
        required=True,
        help='input files of preprocessed dir path')

    parser.add_argument(
        '--is-input-preprocessed',
        type=ast.literal_eval,
        default=True,
        help=
        'input image is preprocessed or not preprocessed. If is preprocessed, each file '
        'each file must be named {name}_{org_h}_{org_w}_{dst_h}_{dst_w}.bin')

    parser.add_argument(
        '--image-count',
        type=int,
        required=False,
        help='image count you want to eval')

    parser.add_argument(
        '--ip', type=str, required=True, help='ip of development board')

    parser.add_argument(
        '--image-type', type=int, required=False, help='image type')

    args = parser.parse_args()

    print(args)
    # TODO random seed
    send_images(args.ip, args.algo_name, args.input_file_path,
                args.is_input_preprocessed, args.image_count, args.image_type)
