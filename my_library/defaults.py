"""
@Time    :2020/2/12 21:17
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

working_dir = '/storage/gs2018/liangjiaxi/bishe'

relation2id = {'主演': 0, '目': 1, '身高': 2, '出生日期': 3, '国籍': 4, '连载网站': 5, '作者': 6, '歌手': 7, '海拔': 8, '出生地': 9, '导演': 10, '气候': 11, '朝代': 12, '妻子': 13, '民族': 14, '毕业院校': 15, '编剧': 16, '出品公司': 17, '父亲': 18, '出版社': 19, '作词': 20, '作曲': 21, '母亲': 22, '成立日期': 23, '字': 24, '丈夫': 25, '号': 26, '所属专辑': 27, '所在城市': 28, '总部地点': 29, '主持人': 30, '上映时间': 31, '首都': 32, '创始人': 33, '祖籍': 34, '改编自': 35, '制片人': 36, '注册资本': 37, '人口数量': 38, '面积': 39, '主角': 40, '占地面积': 41, '嘉宾': 42, '简称': 43, '董事长': 44, '官方语言': 45, '邮政编码': 46, '专业代码': 47, '修业年限': 48}
id2relation = {0: '主演', 1: '目', 2: '身高', 3: '出生日期', 4: '国籍', 5: '连载网站', 6: '作者', 7: '歌手', 8: '海拔', 9: '出生地', 10: '导演', 11: '气候', 12: '朝代', 13: '妻子', 14: '民族', 15: '毕业院校', 16: '编剧', 17: '出品公司', 18: '父亲', 19: '出版社', 20: '作词', 21: '作曲', 22: '母亲', 23: '成立日期', 24: '字', 25: '丈夫', 26: '号', 27: '所属专辑', 28: '所在城市', 29: '总部地点', 30: '主持人', 31: '上映时间', 32: '首都', 33: '创始人', 34: '祖籍', 35: '改编自', 36: '制片人', 37: '注册资本', 38: '人口数量', 39: '面积', 40: '主角', 41: '占地面积', 42: '嘉宾', 43: '简称', 44: '董事长', 45: '官方语言', 46: '邮政编码', 47: '专业代码', 48: '修业年限'}
yuzhi = 0.5
# 设置BERT
unknown = {
    "BertTokenizer": 100
}

