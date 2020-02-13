"""
@Time    :2020/2/12 21:13
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn

主体、客体和关系都是 二元关系
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


@DatasetReader.register("two_stage")
class TwoStageDataReader(DatasetReader):
    def __init__(self,
        pretrained_model_pth: str =None,
        lazy: bool = False,
        mode: str = 's2po'
                 ):
        """
        两阶段数据读取模型
            1、先读s， 再读 o与p
            2、先读o， 再读 s和p
        :param token_indexers:
        :param pretrained_model_pth:
        :param lazy:
        :param mode: 选择模式，先读哪一个
        """
        super().__init__(lazy)
        if pretrained_model_pth is None:
            self._token_indexers ={"tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers ={"tokens": PretrainedTransformerIndexer(pretrained_model_pth)}

        self.pretrained_tokenizer = None
        if pretrained_model_pth is not None:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_pth)
        assert mode in ['s2po', 'o2ps']
        self.mode = mode

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path) # 确保文件路径存在
        with open(file_path) as f:
            for line in f.readlines():
                data = json.loads(line)
                text = data['text']
                spo_list = data['spo_list']
                if self.mode == 's2po':
                    root_leaves: Dict[str: List[Tuple[int, str]]] = build_subject_and_leaves(spo_list)
                elif self.mode == 'o2ps':
                    root_leaves: Dict[str: List[Tuple[int, str]]] = build_object_and_leaves(spo_list)
                else:
                    raise Exception("请检查你的mode参数")
                # 这里的0的意思是代表，将第0行对应的位置置为1
                one: List[Tuple[int, str]] = [(0, w) for w in root_leaves.keys()]
                # 例如，我们将所有的subject的位置都标注出来
                # shape: [seq_len, 1] ; shape: [seq_len, 1]
                one_array_start, one_array_end = get_many_array(text, one, 1)
                for k, lst in root_leaves.items():
                    one_position: Tuple[int, int] = search(text, k)[0]
                    # 如果没有找到k的话，则忽视这条数据。例如没有找到subject，那么这条就可以省略
                    if one_position == (-1, -1):
                        continue
                    many_array_start, many_array_end = get_many_array(text, lst, len(defaults.relation2id))

                    one_array = (one_array_start, one_array_end)
                    many_array = (many_array_start, many_array_end)
                    yield self.text_to_instance(text, one_array, one_position, many_array)

    def text_to_instance(self, text: str, one_array:Tuple[np.array, np.array]=None, one_position: Tuple[int, int]=None, many_array:Tuple[np.array, np.array]=None) -> Instance:
        "训练的时候，输入这些用于训练我们的模型。至于验证时，则应重新写一个验证数据读取类"
        length = len(text)
        if self.pretrained_tokenizer is not None:
            UNK = defaults.unknown[self.pretrained_tokenizer.__class__.__name__]
            tokens = [Token(text=w, text_id=self.pretrained_tokenizer.vocab.get(w, UNK)) for w in text]
        else:
            tokens = [Token(w) for w in text]
        text_field = TextField(tokens, self._token_indexers)


        span = SpanField(one_position[0], one_position[1], text_field)
        dtype:numpy.dtype = np.dtype(numpy.float32)
        one_s = ArrayField(one_array[0], dtype=dtype)
        one_e = ArrayField(one_array[1], dtype=dtype)
        mang_s = ArrayField(many_array[0], dtype=dtype)
        mang_e = ArrayField(many_array[1], dtype=dtype)
        fields = {
            "tokens": text_field,
            "span": span,
            "one_s": one_s,
            "one_e": one_e,
            "many_s": mang_s,
            "many_e": mang_e,
            "metadata": MetadataField(None) # 训练的时候，不需要知道这个。而验证集需要，故占此位置
        }
        return Instance(fields)





