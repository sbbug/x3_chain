# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import argparse
import os
import re
import time
import multiprocessing
from multiprocessing import Process, Queue

import numpy as np
import utils
from data_transformer import data_transformer

from x3_tc_ui import HB_QuantiONNXRuntime
from horizon_nn import horizon_onnx
from coco_metric import MSCOCODetMetric
import sys
sys.path.append("../../../01_common/python/data/")
from transformer import *
from data_loader import *
from x3_tc_ui.utils import tool_utils

CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_PARALLEL_PROCESS_NUM = max(CPU_COUNT - 2, 2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Input detecition onnx model(.onnx) file.",
        required=True)
    parser.add_argument(
        '--image_path',
        type=str,
        help='Evaluate coco data image directory.',
        required=True)
    parser.add_argument(
        '--annotation_path',
        type=str,
        help='Evaluate coco dataset annotation path',
        required=True)
    parser.add_argument(
        '-n',
        '--process-num',
        type=int,
        help='评测使用的进程数',
        default=os.environ.get('PARALLEL_PROCESS_NUM', 5))
    args = parser.parse_args()
    return args


def yolo_loader(image_path, annotation_path):
    transformers = data_transformer()
    transformers.append(ColorConvertTransformer('RGB', 'YUV444'))

    data_loader = COCOValidDataLoader(
        transformers, image_path, annotation_path,batch_size=1, imread_mode='opencv')
    return data_loader


class DetectionExecutor():
    def __init__(self, model, image_path, annotation_path):
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
        self.sess = HB_QuantiONNXRuntime(onnx_model=model)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = [output.name for output in self.sess.get_outputs()]

        self.image_path = image_path
        self.annotation_path = annotation_path

        self.classes = utils.get_classes()
        self.num_classes = len(self.classes)
        self.anchors = np.array([
            10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
            373, 326
        ]).reshape((3, 3, 2))
        self.num_anchors = self.anchors.shape[0]
        self.strides = np.array([8, 16, 32])
        self.input_shape = (672, 672)
        self.metric = MSCOCODetMetric(self.annotation_path, with_mask=False)

    def evaluate(self, data_loader, num_valid=5000):
        """evalute model in coco validation dataset"""
        for i in range(num_valid):
            if i % 10 == 0:
                print('process: {} / {}'.format(i, num_valid))
            image, entry_dict = next(data_loader)
            image = image.transpose([0, 2, 3, 1])
            anno_info = entry_dict[0]
            origin_shape = anno_info['origin_shape']
            bboxes_pr = self.predict(image, origin_shape)
            pred_result = []
            for one_bbox in bboxes_pr:
                one_result = {'bbox': one_bbox, 'mask': False}
                pred_result.append(one_result)
            self.metric.update(pred_result, anno_info['image_name'])

    def predict(self, image, origin_shape):
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            self.output_name, {self.input_name: image},
            {self.input_name: "yuv444"})

        # print(pred_sbbox.shape,pred_mbbox.shape,pred_lbbox.shape)
        pred_sbbox = pred_sbbox.reshape([1, pred_sbbox.shape[1], pred_sbbox.shape[2], 3,
                                         5 + self.num_classes]).transpose([0, 3, 1, 2, 4])
        pred_mbbox = pred_mbbox.reshape([1, pred_mbbox.shape[1], pred_mbbox.shape[2], 3,
                                         5 + self.num_classes]).transpose([0, 3, 1, 2, 4])
        pred_lbbox = pred_lbbox.reshape([1, pred_lbbox.shape[1], pred_lbbox.shape[2], 3,
                                         5 + self.num_classes]).transpose([0, 3, 1, 2, 4])


        pred_sbbox = utils.yolov5_decoder(pred_sbbox, self.num_anchors,
                                          self.num_classes, self.anchors[0],
                                          self.strides[0])
        pred_mbbox = utils.yolov5_decoder(pred_mbbox, self.num_anchors,
                                          self.num_classes, self.anchors[1],
                                          self.strides[1])
        pred_lbbox = utils.yolov5_decoder(pred_lbbox, self.num_anchors,
                                          self.num_classes, self.anchors[2],
                                          self.strides[2])
        pred_bbox = np.concatenate([
            np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_lbbox, (-1, 5 + self.num_classes))
        ],
                                   axis=0)
        bboxes = utils.postprocess_boxes(
            pred_bbox,
            origin_shape,
            input_shape=self.input_shape,
            score_threshold=0.001)
        nms_bboxes = utils.nms(bboxes, 0.65)

        return nms_bboxes

    def result(self, model_name):
        result_file = model_name + '_result.txt'
        with open(result_file, 'w') as f:
            names, values = self.metric.get()
            for name, value in zip(names, values):
                print(name, value)
                record_string = name + ' ' + value + '\n'
                f.write(record_string)

    def report(self):
        names, values = self.metric.get()
        summary = values[0]
        summary = summary.splitlines()
        pattern = re.compile(r'(IoU.*?) .* (.*)$')
        tool_utils.report_flag_start('MAPPER-EVAL')
        for v in summary[0:2]:
            valid_data = pattern.findall(v)[0]
            print("[%s] = %s" % (valid_data[0], valid_data[1]))
        tool_utils.report_flag_end('MAPPER-EVAL')


class ParalleDetectionExecutor(DetectionExecutor):
    '''
    使用多进程进行评测评测。
    为了避免一次性load图片进内存，使用生产消费FIFO
    '''

    def __init__(self, model, image_path, annotation_path, process_num):
        super(ParalleDetectionExecutor, self).__init__(model, image_path,
                                                       annotation_path)
        self._process_num = process_num
        self.model = model

    def predict_wrapper(self, data_item):
        image, entry_dict = data_item
        image = image.transpose([0, 2, 3, 1])
        anno_info = entry_dict[0]
        origin_shape = anno_info['origin_shape']
        bboxes_pr = self.predict(image, origin_shape)
        pred_result = []
        for one_bbox in bboxes_pr:
            one_result = {'bbox': one_bbox, 'mask': False}
            pred_result.append(one_result)
        return pred_result, anno_info['image_name']

    def _producer(self, data_loader, queue_data):
        '''
        :type queue_result: Queue
        '''
        for data_item in data_loader:
            queue_data.put(data_item)

    def _worker(self, queue_data, queue_result):
        while True:
            data_item = queue_data.get()
            if data_item is None:
                break
            result = self.predict_wrapper(data_item)
            queue_result.put(result)
            # print(result)

    def evaluate(self, data_loader, num_valid=5000):
        queue_data = Queue(self._process_num)
        queue_result = Queue(self._process_num)
        producer = Process(
            target=self._producer, args=[data_loader, queue_data])
        workers = [
            Process(
                target=self._worker,
                args=[queue_data, queue_result],
                name='worker_%d' % i) for i in range(self._process_num)
        ]
        producer.start()
        for w in workers:
            print("start worker: %s" % w.name)
            w.start()

        for i in range(num_valid):
            i += 1
            if i % 10 == 0:
                print('process: {} / {}'.format(i, num_valid))
            result = queue_result.get()
            self.metric.update(result[0], result[1])

        try:
            for w in workers:
                w.terminate()
                w.join()
                print("worker: %s exited." % w.name)
            producer.terminate()
        except Exception as e:
            print("exception while terminating: %s" % e)


def check_numpy():
    """Check whether the numpy version is supported now.
       For the coco is uncompatible with numpy with version higher or equal to 1.18.
       Only numpy version lower than 1.18 will return true, others return false.
    """
    curr_numpy = np.__version__
    curr_np_vlist = curr_numpy.split(".")
    if (int(curr_np_vlist[0]) == 1 and int(curr_np_vlist[1]) >= 18):
        return False
    return True


def main():
    start = time.time()
    if not check_numpy():
        print(
            "Numpy version is unsupported, please downgrade numpy to below version 1.18"
        )
        return
    args = get_args()
    onnx_model = horizon_onnx.load(args.model)
    model_name = args.model.split('/')[-1].split('.')[0]
    data_loader = yolo_loader(args.image_path, args.annotation_path)

    valid_executor = ParalleDetectionExecutor(
        onnx_model, args.image_path, args.annotation_path, args.process_num)
    valid_executor.evaluate(data_loader,num_valid=3354)
    valid_executor.result(model_name)
    valid_executor.report()
    end = time.time()

    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("%02d:%02d:%02d" % (h, m, s))


if __name__ == '__main__':
    main()
