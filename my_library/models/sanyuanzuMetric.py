"""
@Time    :2019/10/2 15:24
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
from tqdm import tqdm
import random
from pprint import pprint
import os
import collections
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

@Metric.register("sanyuanzu")
class SanyuanzuMetric(Metric):
    def __init__(self):
        self.all_right = 0
        self.should_have = 0
        self.predict_num = 0

    def __call__(self, spo_lst, predict_spo_lst):
        assert isinstance(spo_lst, list)
        assert isinstance(predict_spo_lst, list)

        spo_set = {(w['subject'], w['predicate'], w['object']) for w in spo_lst}
        pred_spo_set = {(w['subject'], w['predicate'], w['object']) for w in predict_spo_lst}
        self.all_right += len(spo_set & pred_spo_set)
        self.should_have += len(spo_set)
        self.predict_num += len(pred_spo_set)

    @overrides
    def get_metric(self, reset: bool):

        precision = self.all_right / (self.predict_num + 1e-14)
        recall = self.all_right / (self.should_have + 1e-14)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-14)

        if reset:
            self.reset()
        return {
            'precision' : precision,
            "recall": recall,
            'f1': f1
        }
    @overrides
    def reset(self) -> None:
        self.all_right = 0
        self.should_have = 0
        self.predict_num = 0