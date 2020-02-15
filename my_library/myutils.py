"""
@Time    :2020/2/12 22:11
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
import random
import os
import logging
import collections
from pathlib import Path

from allennlp.data import Token
from allennlp.nn.util import get_range_vector, get_device_of, flatten_and_batch_shift_indices
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple, Optional
import re
import doctest
import torch
import numpy as np

from my_library import defaults
from my_library.defaults import relation2id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def search(text: str, word: str) -> List[Tuple[int, int]]:
    """
    查询第一个，包含第一个 i, j 是左闭右闭
    :param text:
    :param word:
    :return: [(i, j),]
    >>> text = '张三的父亲是李四，张三的*母亲是王五'
    >>> search(text, '张三')
    [(0, 1)]
    >>> search(text, '李四')
    [(6, 7)]
    >>> search(text, '?')
    [(-1, -1)]
    >>> search(text, '*')
    [(12, 12)]
    """
    import re
    temp = re.search(re.escape(word), text) # re.escape 将特殊符号转换
    if temp is None:
        return [(-1, -1), ]
    start = temp.start()
    end = temp.end() - 1
    return [(start, end)]

def set_array_(array: torch.Tensor, start: int, end: int, predicate: int):
    """
    传入矩阵，将对应位置变为1
    :param array:
    :param start:
    :param end:
    :param predicate:
    :return:
    """
    assert 0 <= predicate < array.shape[1], '确保关系是合法的'
    array[start, predicate] = 1
    array[end, predicate] = 1

def build_subject_and_leaves(spo_list: List[Dict])->Dict[str, List[Tuple[int, str]]]:
    """
    构造一个根节点作为subject， 多个节点作为父节点
    :param spo_list:
    :return:
    >>> spo_list = [{"predicate": "作词", "object_type": "人物", "subject_type": "歌曲", "object": "施立", "subject": "关键时刻"}, {"predicate": "所属专辑", "object_type": "音乐专辑", "subject_type": "歌曲", "object": "也许明天", "subject": "关键时刻"}, {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}, {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}]
    >>> build_subject_and_leaves(spo_list)
    {'关键时刻': [(20, '施立'), (27, '也许明天'), (7, '张惠妹'), (21, '张惠妹')]}
    """
    s2p_o = collections.defaultdict(list)
    for dct in spo_list:
        subject = dct['subject']
        object = dct['object']
        predicate = dct['predicate']
        s2p_o[subject].append((relation2id[predicate], object))
    return dict(s2p_o)

def build_object_and_leaves(spo_list: List[Dict])->Dict[str, List[Tuple[int, str]]]:
    """
    与上文正好相反
    :param spo_list:
    :return:
    >>> spo_list = [{"predicate": "作词", "object_type": "人物", "subject_type": "歌曲", "object": "施立", "subject": "关键时刻"}, {"predicate": "所属专辑", "object_type": "音乐专辑", "subject_type": "歌曲", "object": "也许明天", "subject": "关键时刻"}, {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}, {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}]
    >>> build_object_and_leaves(spo_list)
    {'施立': [(20, '关键时刻')], '也许明天': [(27, '关键时刻')], '张惠妹': [(7, '关键时刻'), (21, '关键时刻')]}
    """
    o2p_s = collections.defaultdict(list)
    for dct in spo_list:
        subject = dct['subject']
        object = dct['object']
        predicate = dct['predicate']
        o2p_s[object].append((relation2id[predicate], subject))
    return dict(o2p_s)

def build_three_layer_tree(spo_list: List[Dict], mode):
    """
    >>> spo_list = [{"predicate": "作词", "object_type": "人物", "subject_type": "歌曲", "object": "施立", "subject": "关键时刻"}, {"predicate": "所属专辑", "object_type": "音乐专辑", "subject_type": "歌曲", "object": "也许明天", "subject": "关键时刻"}, {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}, {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}]
    >>> build_three_layer_tree(spo_list, 's')
    defaultdict(<function build_three_layer_tree.<locals>.<lambda> at 0x7f3627733400>, {'关键时刻': defaultdict(<class 'list'>, {'施立': [20], '也许明天': [27], '张惠妹': [7, 21]})})

    >>> spo_list = [{"predicate": "作词", "object_type": "人物", "subject_type": "歌曲", "object": "施立", "subject": "关键时刻"}, {"predicate": "所属专辑", "object_type": "音乐专辑", "subject_type": "歌曲", "object": "也许明天", "subject": "关键时刻"}, {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}, {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}]
    >>> build_three_layer_tree(spo_list, 'o')
     defaultdict(<function build_three_layer_tree.<locals>.<lambda> at 0x7f0a861de598>, {'施立': defaultdict(<class 'list'>, {'关键时刻': [20]}), '也许明天': defaultdict(<class 'list'>, {'关键时刻': [27]}), '张惠妹': defaultdict(<class 'list'>, {'关键时刻': [7, 21]})})
    """
    ret = collections.defaultdict(lambda : collections.defaultdict(list))
    for dct in spo_list:
        subject = dct['subject']
        object = dct['object']
        predicate = dct['predicate']
        if mode == 's':
            ret[subject][object].append(relation2id[predicate])
        elif mode == 'o':
            ret[object][subject].append(relation2id[predicate])
        else:
            raise Exception
    return ret

def build_p_so(spo_list: List[Dict]):
    """
    一阶段模型需要每个 关系 到词对的映射
    >>> spo_list = [{"predicate": "作词", "object_type": "人物", "subject_type": "歌曲", "object": "施立", "subject": "关键时刻"}, {"predicate": "所属专辑", "object_type": "音乐专辑", "subject_type": "歌曲", "object": "也许明天", "subject": "关键时刻"}, {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}, {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张惠妹", "subject": "关键时刻"}]
    >>> build_p_so(spo_list)
    {20: [('关键时刻', '施立')], 27: [('关键时刻', '也许明天')], 7: [('关键时刻', '张惠妹')], 21: [('关键时刻', '张惠妹')]}
    """
    ret = collections.defaultdict(list)
    words = set()

    for dct in spo_list:
        subject = dct['subject']
        object = dct['object']
        predicate = dct['predicate']
        ret[relation2id[predicate]].append((subject, object))
        words.add(subject)
        words.add(object)
    return dict(ret), words


# def get_one_array(text:str, leaf:str)->Tuple[np.array, np.array]:
#     """
#     得到一个向量
#     :param text:
#     :param subject:
#     :return:
#     >>> get_one_array("张三的父亲是李四", '父亲')
#     (array([[0.],
#            [0.],
#            [0.],
#            [1.],
#            [0.],
#            [0.],
#            [0.],
#            [0.]]), array([[0.],
#            [0.],
#            [0.],
#            [0.],
#            [1.],
#            [0.],
#            [0.],
#            [0.]]))
#     """
#     start_array = np.zeros((len(text), 1))
#     end_array = np.zeros((len(text), 1))
#     for start, end in search(text, leaf):
#         start_array[start, 0] = 1
#         end_array[end, 0] = 1
#     return start_array, end_array

def get_many_array(text:str, leaves:List[Tuple[int, str]], dim_size: int)->Tuple[np.array, np.array]:
    """
    得到许多向量
    :param text:
    :param leaves:
    :return:
    >>> get_many_array("张三的父亲是李四", [(0, "张三"), (1, "李四")], 49)
    (array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.]]), array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0.]]))
    """
    start_array = np.zeros((len(text), dim_size))
    end_array = np.zeros((len(text),dim_size))
    for p, o in leaves:
        for start, end in search(text, o):
            start_array[start, p] = 1
            end_array[end, p] = 1
    return start_array, end_array

def get_empty(text, dim)->Tuple[np.array, np.array]:
    "我需要一些空的矩阵"
    start_array = torch.zeros((len(text), dim))
    end_array = torch.zeros((len(text), dim))
    return start_array, end_array

def get_span_list(array1, array2, yuzhi):
    """
    输入 2个一维数组， 得到[[1, 2], [3, 4]] 的列表
    # 就近匹配
    >>> array1 = torch.tensor([0.6,0.3,0.4,0.6,0.1])
    >>> array2 = torch.tensor([0.1,0.6,0.4,0.3,0.6])
    >>> get_span_list(array1, array2, 0.5)
    [[0, 1], [3, 4]]

    # 多个结尾选最近
    >>> array1 = torch.tensor([0.6,0.3,0.4,0.6,0.1])
    >>> array2 = torch.tensor([0.6,0.6,0.4,0.3,0.6])
    >>> get_span_list(array1, array2, 0.5)
    [[0, 0], [3, 4]]

    # 只有一个
    >>> array1 = torch.tensor([0.3,0.3,0.4,0.6,0.1])
    >>> array2 = torch.tensor([0.1,0.3,0.4,0.6,0.3])
    >>> get_span_list(array1, array2, 0.5)
    [[3, 3]]

    # 多个开始选最近
    >>> array1 = torch.tensor([0.6,0.6,0.6,0.6,0.6])
    >>> array2 = torch.tensor([0.1,0.6,0.4,0.3,0.6])
    >>> get_span_list(array1, array2, 0.5)
    [[0, 1], [1, 1], [2, 4], [3, 4], [4, 4]]
    """
    assert array1.dim() == 1 and array2.dim() == 1, f"{array1.shape} {array2.shape}"
    # 返回的是 二元组， （torch([]), dtype =...）
    start_index = torch.where(array1 > yuzhi)[0]
    end_index = torch.where(array2 > yuzhi)[0]

    ans = []
    # 左闭右闭
    for i in start_index:
        j = end_index[end_index >= i]
        if len(j) > 0:
            # 最近匹配，可以根据你的论文选择最近、最远等等
            j = j[0]
            ans.append([i.item(), j.item()])
    return ans

def get_word_from_pretrained(pretrained_tokenizer, text):
    UNK = defaults.unknown[pretrained_tokenizer.__class__.__name__]
    tokens = []
    for t in text:
        idx = pretrained_tokenizer.vocab.get(t, None)
        if idx is None:
            logger.info(f"{t} 不在词表中，过多情况请检查！！如果是训练阶段，可以执行cibiao_sub.py将bert模型中的词表进行替换")
            idx = UNK
        tokens.append(Token(text=t, text_id=idx))
    return tokens

#TODO
def matrix_decode(m_s:torch.Tensor, m_e: torch.Tensor, length):
    if torch.is_tensor(length):
        length = length.item()
    start_tuple = torch.where(m_s > defaults.yuzhi)
    end_


if __name__ == '__main__':
    # doctest.testmod(name= 'build_object_and_leaves')
    doctest.testmod(name = build_p_so.__name__)