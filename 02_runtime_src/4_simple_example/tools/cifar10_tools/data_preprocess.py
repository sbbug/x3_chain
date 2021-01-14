# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import sys
from absl import flags
from absl import logging
from absl import app

import numpy as np
from data_transformer import data_transformer
from easydict import EasyDict
import skimage.io

from easydict import EasyDict
from horizon_nn.data.data_loader import *
from loader import CifarLoader

sys.path.append('.')

FLAGS = flags.FLAGS
flags.DEFINE_string('src_dir', default=None, help='Input image file.')
flags.DEFINE_string('dst_dir', default=None, help='Output bgr dir.')

transformers = data_transformer()


def dataset_loader(cifar_path=None, batch_size=1):
    loader = DataLoader(CifarLoader(cifar_path), transformers, batch_size)
    return loader


def process_file(src_file, data_loader, dst_dir):
    # fw = open("./val.txt", 'w')  # 将要输出保存的文件地址
    for i in range(10000):
        image, label = next(data_loader)
        image_name = str(i) + ".jpg"
        org_h = 32
        org_w = 32
        dst_h = 32
        dst_w = 32
        file_name = '{}_{}_{}_{}_{}.bin'.format(image_name, org_h, org_w,
                                                dst_h, dst_w)
        pic_name = os.path.join(dst_dir + '/' + file_name)
        logging.info("write:%s" % pic_name)
        # fw.write(str(i) + ".jpg " + str(label[0]))  # 将字符串写入文件中
        # fw.write("\n")  # 换行
        image[0].astype(np.uint8).tofile(pic_name)
        # print(image)
        # print(image.shape)


def main(_):
    src_dir = FLAGS.src_dir
    dst_dir = FLAGS.dst_dir
    data_loader = dataset_loader(FLAGS.src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    process_file(src_dir, data_loader, dst_dir)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
