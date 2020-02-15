"""
@Time    :2020/2/14 21:13
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json

import numpy
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField, SpanField, ArrayField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from overrides import overrides
from transformers import AutoTokenizer

from tqdm import tqdm
import random
from pprint import pprint
import os
import collections
from typing import List, Dict, Tuple, Iterable
import logging
from pathlib import Path

from my_library import defaults
from my_library.myutils import *


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



@DatasetReader.register("one_stage")
class OneStageDataReader(DatasetReader):
    def __init__(self,
                 pretrained_model_pth: str =None,
                 lazy: bool = False,
                 ):
        """
        一阶段模型
            同时得到所有的关系。模型是
            解码可能会比较麻烦
            matrix attention
        :param token_indexers:
        :param pretrained_model_pth:
        :param lazy:
        """
        super().__init__(lazy)
        if pretrained_model_pth is None:
            self._token_indexers ={"tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers ={"tokens": PretrainedTransformerIndexer(pretrained_model_pth)}

        self.pretrained_tokenizer = None
        if pretrained_model_pth is not None:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_pth)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path) # 确保文件路径存在
        with open(file_path) as f:
            for line in f.readlines():
                data = json.loads(line)
                text = data['text'].replace(" ", "")
                spo_list = data['spo_list']
                root_leaves, words = build_p_so(spo_list)
                # shape: [seq_len, seq_len, relationship]
                target_matrix_start = torch.zeros((len(text), len(text), len(relation2id))).cpu().numpy()
                target_matrix_end = torch.zeros((len(text), len(text), len(relation2id))).cpu().numpy()
                position = {}
                for word in words:
                    position[word] = search(text, word)[0]
                for r_id, lst in root_leaves.items():
                    for s, o in lst:
                        s_span = position[s]
                        o_span = position[o]
                        x_1, y_1 = s_span[0], o_span[0]
                        x_2, y_2 = s_span[1], o_span[1]
                        target_matrix_start[x_1, y_1, r_id] = 1
                        target_matrix_end[x_2, y_2, r_id] = 1

                yield self.text_to_instance(text, target_matrix_start, target_matrix_end)

    def text_to_instance(self, text: str, target_matrix_start: np.array, target_matrix_end: np.array) -> Instance:
        "训练的时候，输入这些用于训练我们的模型。至于验证时，则应重新写一个验证数据读取类"
        if self.pretrained_tokenizer is not None:
            tokens = get_word_from_pretrained(self.pretrained_tokenizer, text)
        else:
            tokens = [Token(w) for w in text]
        text_field = TextField(tokens, self._token_indexers)

        fields = {
            "tokens": text_field,
            "target_start": ArrayField(target_matrix_start),
            "target_end": ArrayField(target_matrix_end),
            "metadata": MetadataField(None) # 训练的时候，不需要知道这个。而验证集需要，故占此位置
        }
        return Instance(fields)


