"""
@Time    :2020/2/15 10:06
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

# def run(exp_name):
#     for config_file in os.listdir(f" ./configs/{exp_name}"):
#         print(os.getcwd())
#         command = f"allennlp train ./configs/{exp_name}/{config_file}.json -s ./output/{exp_name}/{config_file.split('.')[0]}   -f   --include-package my_library"
#         sys.argv = command.split()
#         print(sys.argv)
#         main()
# if __name__ == '__main__':
#     fire.Fire(run)

def run(exp_name):
    for config_file in [
        "r_lstm_2_fcn_1",
        "r_lstm_2_fcn_1_300",
        "w_lstm_2_fcn_1_300",
        "bert_2_fcn_1",
        "bert_6_fcn_1",
    ]:
        print(os.getcwd())
        command = f"allennlp train ./configs/{exp_name}/{config_file}.json -s ./output/{exp_name}/{config_file.split('.')[0]}   -f   --include-package my_library"
        sys.argv = command.split()
        print(sys.argv)
        main()
if __name__ == '__main__':
    run('exp1')