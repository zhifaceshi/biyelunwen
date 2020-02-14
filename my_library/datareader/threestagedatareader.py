"""
@Time    :2020/2/13 21:12
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""

import numpy
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField, SpanField, ArrayField, MetadataField, LabelField
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

from my_library.myutils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@DatasetReader.register("three_stage")
class ThreeStageDataReader(DatasetReader):
    def __init__(self,
                 pretrained_model_pth: str =None,
                 lazy: bool = False,
                 mode: str = 'sop'
                 ):
        """
        三阶段数据读取模型
            1、先读s， 再读o，最后读pos
            2、先读o， 再读s，最后读p
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
        assert mode in ['sop', 'osp']
        self.mode = mode

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path) # 确保文件路径存在
        with open(file_path) as f:
            for line in f.readlines():
                data = json.loads(line)
                text = data['text'].replace(" ", "")
                spo_list = data['spo_list']
                if self.mode == 'sop':
                    root_leaves: Dict[str: List[Tuple[int, str]]] = build_three_layer_tree(spo_list, 's')
                elif self.mode == 'osp':
                    root_leaves: Dict[str: List[Tuple[int, str]]] = build_three_layer_tree(spo_list, 'o')
                else:
                    raise Exception("请检查你的mode参数")
                # 这里的0的意思是代表，将第0行对应的位置置为1
                one: List[Tuple[int, str]] = [(0, w) for w in root_leaves.keys()]
                # 例如，我们将所有的subject的位置都标注出来
                # shape: [seq_len, 1] ; shape: [seq_len, 1]
                one_array_start, one_array_end = get_many_array(text, one, 1)
                one_array = (one_array_start, one_array_end)
                for k, dct in root_leaves.items():
                    one_position = search(text, k)[0]
                    if one_position == (-1, -1):
                        continue
                    two = [(0, w) for w in dct.keys()]
                    # shape: [seq_len, 1] ; shape: [seq_len, 1]
                    two_array_start, two_array_end = get_many_array(text, two, 1)
                    two_array = (two_array_start, two_array_end)
                    for v, relation2id in dct.items():
                        two_position = search(text, v)[0]
                        if two_position == (-1, -1):
                            continue
                        # 可能存在一对实体存在多个关系
                        relationship_array = [0 for _ in range(len(defaults.id2relation))]
                        for i in relation2id:
                            relationship_array[i] = 1
                        relationship_array = numpy.array(relationship_array)
                        yield self.text_to_instance(
                            text,
                            one_array,
                            two_array,
                            one_position,
                            two_position,
                            relationship_array
                        )

    def text_to_instance(self, text: str, one_array:Tuple[np.array, np.array]=None,
                         two_array:Tuple[np.array, np.array]=None,
                         one_position: Tuple[int, int]=None,
                         two_position: Tuple[int, int]=None,
                         relation2id: np.array = None,
                         ) -> Instance:
        "训练的时候，输入这些用于训练我们的模型。至于验证时，则应重新写一个验证数据读取类"
        if self.pretrained_tokenizer is not None:
            tokens = get_word_from_pretrained(self.pretrained_tokenizer, text)
        else:
            tokens = [Token(w) for w in text]
        text_field = TextField(tokens, self._token_indexers)

        one_span = SpanField(one_position[0], one_position[1], text_field)
        two_span = SpanField(two_position[0], two_position[1], text_field)

        dtype:numpy.dtype = np.dtype(numpy.float32)
        one_s = ArrayField(one_array[0], dtype=dtype)
        one_e = ArrayField(one_array[1], dtype=dtype)
        two_s = ArrayField(two_array[0], dtype=dtype)
        two_e = ArrayField(two_array[1], dtype=dtype)
        relation2id = ArrayField(relation2id, dtype=dtype)
        fields = {
            "tokens": text_field,
            "one_span": one_span,
            "two_span": two_span,
            "one_s": one_s,
            "one_e": one_e,
            "two_s": two_s,
            "two_e": two_e,
            "relationship": relation2id,
            "metadata": MetadataField(None) # 训练的时候，不需要知道这个。而验证集需要，故占此位置
        }
        return Instance(fields)


