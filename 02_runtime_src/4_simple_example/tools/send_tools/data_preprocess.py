# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
from absl import flags
from absl import logging
from absl import app

import numpy as np
from pathos import multiprocessing
import traceback

from data_loader import image_loader

FLAGS = flags.FLAGS

CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_PARALLEL_PROCESS_NUM = max(CPU_COUNT - 2, 2)

flags.DEFINE_string('input_file_path', None, help='Input images dir')
flags.DEFINE_integer('image_count', 0, help='Image count to be processed')
flags.DEFINE_string(
    'algo_name',
    None,
    help='model type, such as mobilenetv1, mobilenetv2, yolov2, yolov3')
flags.DEFINE_string(
    'output_file_path',
    None,
    help='Output dir, output file will stored as '
    '{output_file_path}/{name}_{org_h}_{org_w}_{result_h}_{result_w}.bin')
flags.DEFINE_integer('image_type', 2, help='Image type to be processed to')
flags.DEFINE_integer(
    'parallel_num',
    os.environ.get('PARALLEL_PROCESS_NUM', 1),
    help='Parallel process number')


def process_image(image_dir, image_name, algo_name, image_type, output_dir):
    try:
        path = os.path.join(image_dir, image_name)

        org_h, org_w, dst_h, dst_w, data = image_loader(
            path, algo_name, image_type)
        file_name = '{}_{}_{}_{}_{}.bin'.format(image_name, org_h, org_w,
                                                dst_h, dst_w)
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            return
        with open(output_file, 'wb') as w:
            w.write(data)
        logging.info("Process {}, saved as {}".format(path, output_file))
    except Exception as e:
        logging.info(traceback.format_exc())
        raise e


def preprocess_images(algo_name, input_file_path, image_count, image_type,
                      output_file_path, parallel_num):
    files = os.listdir(input_file_path)
    if not image_count:
        image_count = len(files)
    elif image_count > len(files):
        logging.fatal('image count is too large')
    os.makedirs(output_file_path, exist_ok=True)

    processed_files = set()
    for f in os.listdir(output_file_path):
        processed_files.add('_'.join(f.split('_')[:-4]))

    pool = multiprocessing.Pool(processes=parallel_num, )
    pool._taskqueue.maxsize = parallel_num  # hack here

    results = []
    for i in np.arange(image_count):
        path = os.path.join(input_file_path, files[i])
        if os.path.isdir(path):
            logging.fatal("Input path contains a dir !!!")

        if files[i] in processed_files:
            logging.warning("{} already preprocessed, ignore".format(path))
            continue

        args = (input_file_path, files[i], algo_name, image_type,
                output_file_path)
        r = pool.apply_async(
            func=process_image, error_callback=logging.error, args=args)
        results.append(r)

    for r in results:
        r.get()
    pool.close()
    pool.join()


def main(_):
    logging.set_verbosity(logging.INFO)
    preprocess_images(FLAGS.algo_name, FLAGS.input_file_path,
                      FLAGS.image_count, FLAGS.image_type,
                      FLAGS.output_file_path, FLAGS.parallel_num)


if __name__ == "__main__":
    app.run(main)
