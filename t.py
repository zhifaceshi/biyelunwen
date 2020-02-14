# """
# @Time    :2020/2/13 15:25
# @Author  : 梁家熙
# @Email:  :11849322@mail.sustech.edu.cn
# """
# import json
# import random
# import os
# import logging
# import collections
# from pathlib import Path
# from tqdm import tqdm
# from pprint import pprint
# from typing import List, Dict, Tuple
#
from transformers import AutoTokenizer, AutoModel
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# O - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json not fo
# - INFO - https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt not found i


from allennlp.data.token_indexers import PretrainedTransformerIndexer
# pretrained_model_pth =  '/storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/Bert/bert-base-chinese-vocab.txt'
# AutoTokenizer.from_pretrained(pretrained_model_pth)
model_name = '/storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/bert/'
# PretrainedTransformerIndexer(model_name)
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
