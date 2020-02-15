"""
@Time    :2020/2/12 21:04
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
import fire
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
from pathlib import Path
from allennlp.commands import main

override_dict = {"train_data_path":"/storage/gs2018/liangjiaxi/bishe/data/processed_data/fake_train.txt","validation_data_path": "/storage/gs2018/liangjiaxi/bishe/data/processed_data/fake_dev.txt","test_data_path":"/storage/gs2018/liangjiaxi/bishe/data/processed_data/fake_test.txt"}
override_dict = str(override_dict).replace(' ', "")
def run(exp_name, config_file, ):
    print(os.getcwd())
    command = f"allennlp train ./configs/{exp_name}/{config_file}.json -s ./output/{exp_name}/{config_file.split('.')[0]}  -f  -o {override_dict}    --include-package my_library"
    print(sys.argv)
    print(command)
    sys.argv = command.split()

    main()
if __name__ == '__main__':
    exp_name = "exp1"
    config_file = 'bert_fcn_1'
    run(exp_name, config_file)