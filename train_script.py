"""
@Time    :2020/2/14 12:00
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

def run(exp_name, config_file, qianru = '--force'):
    # if qianru == 'r':
    #     qianru = '--recover'
    # elif qianru == 'f':
    #     qianru = '--force'
    # else:
    #     raise Exception
    print(os.getcwd())
    command = f"allennlp train ./configs/{exp_name}/{config_file}.json -s ./output/{exp_name}/{config_file.split('.')[0]}   {qianru}   --include-package my_library"
    sys.argv = command.split()
    print(sys.argv)
    main()
if __name__ == '__main__':
    fire.Fire(run)