"""
@Time    :2020/2/13 9:09
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
import random
import os
import logging
import collections
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple, Iterable
import numpy
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField, SpanField, ArrayField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides
from transformers import AutoTokenizer

from my_library import defaults

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#验证集的数据读取类，应该是通用的
@DatasetReader.register('valid')
class ValidatingDataReader(DatasetReader):
    def __init__(self,
                 pretrained_model_pth: str =None,
                 ):
        """
        两阶段数据读取模型
            1、先读s， 再读 o与p
            2、先读o， 再读 s和p
        :param token_indexers:
        :param pretrained_model_pth:
        :param lazy:
        """
        super().__init__(False)
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

        self.pretrained_tokenizer = None
        if pretrained_model_pth is not None:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_pth)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path) # 确保文件路径存在
        with open(file_path) as f:
            for line in f.readlines():
                data = json.loads(line)
                text = data['text']
                spo_list = data['spo_list']
                yield self.text_to_instance(text, spo_list)

    def text_to_instance(self, text: str, spo_list: List[Dict[str, str]]=None) -> Instance:
        if self.pretrained_tokenizer is not None:
            UNK = defaults.unknown[self.pretrained_tokenizer.__class__.__name__]
            tokens = [Token(text=w, text_id=self.pretrained_tokenizer.vocab.get(w, UNK)) for w in text]
        else:
            tokens = [Token(w) for w in text]
        text_field = TextField(tokens, self._token_indexers)
        fields = {
            'tokens': text_field,
            'metadata': MetadataField({'text': text, 'spo_list': spo_list})
        }
        return Instance(fields)