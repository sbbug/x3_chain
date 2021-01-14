# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import json
from absl import flags
from absl import logging
from absl import app
from x3_tc_ui.utils import tool_utils

flags.DEFINE_string(
    'annotation_path', None, help='Evaluate data label val.txt path')

flags.DEFINE_string("eval_result_path", None, help="Eval result file path")
FLAGS = flags.FLAGS


def parse_val_annotation(filename):
    with open(filename) as f:
        labels = {}
        for line in f.readlines():
            fields = line.strip().split(' ')
            labels[fields[0]] = int(fields[1])

    return labels


def parse_predict_result(filename):
    with open(filename) as f:
        labels = {}
        for line in f.readlines():
            data = json.loads(line.strip())
            image_name = data['frame']['image_name']
            label = data['result'][0]['id']
            labels[image_name] = label
    return labels


def _report(result):
    ctx = os.getenv('HORIZON_RUN_CONTEXT', '')
    print()
    tool_utils.report_flag_start('RUNTIME-%s-EVAL' % ctx)
    print(result)
    tool_utils.report_flag_end('RUNTIME-%s-EVAL' % ctx)


def main(_):
    val_labels = parse_val_annotation(FLAGS.annotation_path)
    pred_labels = parse_predict_result(FLAGS.eval_result_path)

    val_count = len(val_labels)
    pred_count = len(pred_labels)
    if pred_count < val_count:
        logging.warning("Partial predicted, {}/{}".format(
            pred_count, val_count))
    total_count = 0
    correct_count = 0
    for image_name, pred_label in pred_labels.items():
        if image_name in val_labels:
            total_count += 1
            if pred_labels[image_name] == val_labels[image_name]:
                correct_count += 1

    acc = correct_count / (total_count or 1)
    result = 'Predicted:{}/{} Accuracy: {:.4f}'.format(total_count, val_count,
                                                       acc)
    logging.info(result)
    with open('result.txt', 'w') as f:
        f.write(result)
    _report(result)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
