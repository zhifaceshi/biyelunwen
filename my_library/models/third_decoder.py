"""
@Time    :2020/2/17 16:15
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
import random
import os
import logging
import collections
from pathlib import Path

from allennlp.modules import Seq2VecEncoder, FeedForward
from allennlp.nn import Activation
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@Seq2VecEncoder.register("my_seq2vec")
class ThirdPart(Seq2VecEncoder):
    def __init__(self, seq: Seq2VecEncoder,
                 ):
        super().__init__()
        self.seq = seq
        self.feedforward = FeedForward(self.seq.get_output_dim(), 1, 49, Activation.by_name('linear')())
    def forward(self, tokens, mask):
        batchsize, seq_len, dim_size = tokens.shape
        seq = self.seq(tokens, mask)
        assert len(seq.shape) == 2, f"{seq.shape}"
        return self.feedforward(seq)