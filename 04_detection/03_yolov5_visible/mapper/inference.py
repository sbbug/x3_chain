# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import cv2
import os
import numpy as np
from x3_tc_ui import HB_QuantiONNXRuntime
import argparse

from horizon_nn import horizon_onnx
from data_transformer import data_transformer
import utils
import sys

sys.path.append("../../../01_common/python/data/")
from transformer import *
from data_loader import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default="./model_output/yolov5_quantized_model.onnx",
        help='input onnx model(.onnx) file',
        required=False)
    parser.add_argument(
        '--visible',
        type=str,
        default="../test_img/I03181.jpg",
        dest='visible',
        help='input image file',
        required=False)
    args = parser.parse_args()
    return args


def load_image(image_file):
    transformers = data_transformer()
    transformers.append(ColorConvertTransformer('RGB', 'YUV444'))
    origin_image, process_image = DetectionLoader(
        transformers, image_file, imread_mode='opencv')
    return origin_image, process_image


def onnx_forward(sess, image_data):
    input_name = sess.get_inputs()[0].name
    # print(input_name,sess.get_inputs()[0].type,type(image_data),type(image_data[0][0][0][0]))
    # image_data = image_data.astype(np.int8)
    # print(input_name, sess.get_inputs()[0].type, type(image_data), type(image_data[0][0][0][0]))
    output_name = [output.name for output in sess.get_outputs()]
    pred_lbbox, pred_mbbox, pred_sbbox = sess.run(
        output_name, {input_name: image_data}, {input_name: "yuv444"})
    model_output = sess.run(output_name, {input_name: image_data},
                            {input_name: "yuv444"})
    # output_dict = {}
    # for output, name in zip(model_output, output_name):
    #     output_dict[name] = output
    return model_output


def sample(onnx_model, origin_image, process_image):
    strides = np.array([8, 16, 32])
    anchors = np.array([
        10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
        373, 326
    ]).reshape((3, 3, 2))
    num_anchors = anchors.shape[0]
    classes = utils.get_classes()
    num_classes = len(classes)
    _, org_height, org_width, __ = origin_image.shape
    _, __, process_height, process_width = process_image.shape
    process_image = process_image.transpose([0, 2, 3, 1])

    sess = HB_QuantiONNXRuntime(onnx_model=onnx_model)
    outputs = onnx_forward(sess, process_image)
    # print(outputs)
    print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    outputs[0] = outputs[0].reshape([1, outputs[0].shape[1], outputs[0].shape[2], 3,
                                     5 + num_classes]).transpose([0, 3, 1, 2, 4])
    outputs[1] = outputs[1].reshape([1, outputs[1].shape[1], outputs[1].shape[2], 3,
                                     5 + num_classes]).transpose([0, 3, 1, 2, 4])
    outputs[2] = outputs[2].reshape([1, outputs[2].shape[1], outputs[2].shape[2], 3,
                                     5 + num_classes]).transpose([0, 3, 1, 2, 4])

    pred_sbbox, pred_mbbox, pred_lbbox = outputs[0], outputs[1], outputs[2]
    pred_sbbox = utils.yolov5_decoder(pred_sbbox, num_anchors, num_classes,
                                      anchors[0], strides[0])
    pred_mbbox = utils.yolov5_decoder(pred_mbbox, num_anchors, num_classes,
                                      anchors[1], strides[1])
    pred_lbbox = utils.yolov5_decoder(pred_lbbox, num_anchors, num_classes,
                                      anchors[2], strides[2])
    pred_bbox = np.concatenate([
        np.reshape(pred_sbbox, (-1, 5 + num_classes)),
        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
        np.reshape(pred_lbbox, (-1, 5 + num_classes))
    ],
        axis=0)

    bboxes = utils.postprocess_boxes(
        pred_bbox, (org_height, org_width),
        input_shape=(process_height, process_width),
        score_threshold=0.3)

    nms_bboxes = utils.nms(bboxes, 0.65)

    utils.draw_bboxs(origin_image[0], nms_bboxes)


def main():
    args = get_args()
    onnx_model = horizon_onnx.load_model(args.model)

    origin_image, process_image = load_image(args.visible)
    sample(onnx_model, origin_image, process_image)


if __name__ == '__main__':
    main()
