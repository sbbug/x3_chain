# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from easydict import EasyDict
import numpy as np

yolov2_config = EasyDict()

yolov2_config.ANCHORS = np.array([
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778,
    9.77052, 9.16828
]).reshape((5, 2))
yolov2_config.STRIDES = 32

yolov3_config = EasyDict()
yolov3_config.ANCHORS = np.array([
    1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875,
    7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875
]).reshape((3, 3, 2))
yolov3_config.STRIDES = np.array([8, 16, 32])

coco_config = EasyDict()
coco_config.NUM_CLASSES = 10
coco_config.CLASSES = [
    'pedestrian', 'people',
     'bicycle', 'car', 'van', 'truck',
     'tricycle', 'awning-tricycle', 'bus',
     'motor'
]
# coco_config.NUM_CLASSES = 20
# coco_config.CLASSES = [
#     "Expressway-Service-area", "Expressway-toll-station", "airplane",
#     "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
#     "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
#     "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
# ]
