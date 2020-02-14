"""
@Time    :2020/2/14 18:36
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
修改BERT的词表，使BERT的未登录词现象大幅降低

"""
import json
import random
import os
import logging
import collections
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def check(line):
    if line.startswith("[un"):
        return True
    if line.startswith('##'):
        return True
    return False
def sub(data_file, vocab_file, output_file):

    with open(vocab_file) as f:
        lines = [w.strip() for w in f.readlines()]
        length = len(lines)
        vocab_char = set(lines)
    char_set = set()
    for pth in  data_file:
        with open(pth) as f:
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                text = data['text']
                char_set = char_set.union(set(text))
    logger.info(f"charset is {len(char_set)}")
    oov = list(char_set - vocab_char)
    logger.info(f"oov is {len(oov)}")

    i = 0
    j = 0
    while i < len(oov) and j < len(lines):
        while not check(lines[j]):
            j += 1
        lines[j] = oov[i]
        i += 1
        j += 1
    assert length == len(lines)
    logger.info(f"lines {len(lines)}")
    assert len(char_set - set(lines) ) == 0
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
if __name__ == '__main__':
    data_file = ["/storage/gs2018/liangjiaxi/bishe/data/processed_data/_train.txt",'/storage/gs2018/liangjiaxi/bishe/data/processed_data/_dev.txt','/storage/gs2018/liangjiaxi/bishe/data/processed_data/_test.txt']
    vocab_file = '/storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/bert/bert-base-chinese-vocab.txt'
    output_file = '/storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/bert/vocab.txt'
    sub(data_file, vocab_file, output_file)


