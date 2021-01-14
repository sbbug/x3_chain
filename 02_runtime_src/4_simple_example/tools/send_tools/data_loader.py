# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import numpy as np
import skimage.io
import cv2
from horizon_nn.data.transformer import *


def get_transformers_by_algo_name(algo_name, image_type):
    # F32 float32 10
    if image_type == 10:
        if algo_name == "googlenet":
            means = np.array([103.94, 116.78, 123.68], dtype=np.float32)
            transformers = [
                ResizeTransformer((224, 224)),
                ScaleTransformer(255),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                MeanTransformer(means),
            ]
            return transformers, 224, 224

        if algo_name == "mobilenetv1" or algo_name == "mobilenetv2":
            means = np.array([103.94, 116.78, 123.68], dtype=np.float32)
            transformers = [
                LongSideCropTransformer(),
                ResizeTransformer((224, 224)),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(255),
                MeanTransformer(means),
                ScaleTransformer(0.017),
            ]
            return transformers, 224, 224

        if algo_name == "yolov2":
            transformers = [
                PadResizeTransformer((608, 608)),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(1 / 255),
            ]
            return transformers, 608, 608

        if algo_name == "yolov3":
            transformers = [
                PadResizeTransformer((416, 416)),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(1 / 255),
            ]
            return transformers, 416, 416

        if algo_name == "resnet50":
            means = np.array([103.94, 116.78, 123.68], dtype=np.float32)
            transformers = [
                ShortSideResizeTransformer(),
                CenterCropTransformer(),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(255),
                MeanTransformer(means),
            ]
            return transformers, 224, 224

        if algo_name == "resnet18":
            means = np.array([103.94, 116.78, 123.68], dtype=np.float32)
            transformers = [
                ShortSideResizeTransformer(),
                CenterCropTransformer(),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(255),
                MeanTransformer(means),
            ]
            return transformers, 224, 224


    if image_type == 3:
        if algo_name == "lenet_onnx" or algo_name == 'lenet':
            transformers = [
                ResizeTransformer((32, 32)),
                TransposeTransformer((2, 0, 1)),

            ]
            return transformers, 32, 32
    if image_type == 2:
        if algo_name == "lenet_onnx" or algo_name == 'lenet':
            transformers = [
                ResizeTransformer((32, 32)),
                TransposeTransformer((2, 0, 1)),

            ]
            return transformers, 32, 32
        if algo_name == "googlenet":
            transformers = [
                ResizeTransformer((224, 224)),
                ScaleTransformer(255),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ColorConvertTransformer('BGR', 'YUV444'),
            ]
            return transformers, 224, 224

        if algo_name == "mobilenetv1" or algo_name == "mobilenetv2":
            transformers = [
                ShortSideResizeTransformer(),
                CenterCropTransformer(),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(255),
                ColorConvertTransformer('BGR', 'YUV444'),
            ]
            return transformers, 224, 224

        if algo_name == "mobilenetv2_onnx":
            transformers = [
                ShortSideResizeTransformer(),
                CenterCropTransformer(),
                TransposeTransformer((2, 0, 1)),
                ScaleTransformer(255),
                ColorConvertTransformer('BGR', 'YUV444'),
            ]
            return transformers, 224, 224

        if algo_name == "yolov2":
            transformers = [
                PadResizeTransformer((608, 608)),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, 608, 608

        if algo_name == "yolov3" or algo_name =="yolov3_onnx":
            transformers = [
                PadResizeTransformer((608, 608)),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, 608, 608

        if algo_name == "yolov5":
            transformers = [
                PadResizeTransformer((672, 672)),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, 672, 672

        if algo_name == "resnet50":
            transformers = [
                ShortSideResizeTransformer(),
                CenterCropTransformer(),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(255),
                ColorConvertTransformer('BGR', 'YUV444'),
            ]
            return transformers, 224, 224

        if algo_name == "resnet18":
            transformers = [
                ShortSideResizeTransformer(),
                CenterCropTransformer(),
                TransposeTransformer((2, 0, 1)),
                ChannelSwapTransformer((2, 1, 0)),
                ScaleTransformer(255),
                ColorConvertTransformer('BGR', 'YUV444'),
            ]
            return transformers, 224, 224

        if algo_name == "efficientnet_lite0":
            image_size = 224
            transformers = [
                PaddedCenterCropTransformer(image_size, 32),
                ResizeTransformer((image_size, image_size), mode='skimage',
                                  method=3),
                TransposeTransformer((2, 0, 1)),
                ScaleTransformer(255),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, image_size, image_size

        if algo_name == "efficientnet_lite1":
            image_size = 240
            transformers = [
                PaddedCenterCropTransformer(image_size, 32),
                ResizeTransformer((image_size, image_size), mode='skimage',
                                  method=3),
                TransposeTransformer((2, 0, 1)),
                ScaleTransformer(255),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, image_size, image_size

        if algo_name == "efficientnet_lite2":
            image_size = 260
            transformers = [
                PaddedCenterCropTransformer(image_size, 32),
                ResizeTransformer((image_size, image_size), mode='skimage',
                                  method=3),
                TransposeTransformer((2, 0, 1)),
                ScaleTransformer(255),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, image_size, image_size

        if algo_name == "efficientnet_lite3":
            image_size = 280
            transformers = [
                PaddedCenterCropTransformer(image_size, 32),
                ResizeTransformer((image_size, image_size), mode='skimage',
                                  method=3),
                TransposeTransformer((2, 0, 1)),
                ScaleTransformer(255),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, image_size, image_size

        if algo_name == "efficientnet_lite4":
            image_size = 300
            transformers = [
                PaddedCenterCropTransformer(image_size, 32),
                ResizeTransformer((image_size, image_size), mode='skimage',
                                  method=3),
                TransposeTransformer((2, 0, 1)),
                ScaleTransformer(255),
                ColorConvertTransformer('RGB', 'YUV444'),
            ]
            return transformers, image_size, image_size


def image_loader(image_path, algo_name, image_type):
    def image_load_func(x):
        return skimage.img_as_float(
            skimage.io.imread(x)).astype(np.float32)

    count = 0

    transformers, dst_h, dst_w \
        = get_transformers_by_algo_name(algo_name, image_type)

    if algo_name == "yolov2" or algo_name == "yolov3" or algo_name == "yolov5" or algo_name == 'lenet_onnx' or algo_name == 'lenet':
        image_read_method = cv2.imread
    else:
        image_read_method = image_load_func

    image = image_read_method(image_path).astype(np.float32)
    org_height = image.shape[0]
    org_width = image.shape[1]

    if image.ndim != 3:  # expend gray scale image to three channels
        image = image[..., np.newaxis]
        image = np.concatenate([image, image, image], axis=-1)

    image = [image]
    for tran in transformers:
        image = tran(image)

    image = image[0].transpose((1, 2, 0)).reshape(-1).astype(np.uint8)
    return org_height, org_width, dst_h, dst_w, image
