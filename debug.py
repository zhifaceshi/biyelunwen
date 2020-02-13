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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
from allennlp.commands import main
print(os.getcwd())
command = "allennlp train ./configs/debug2.json -s ./output/debug -f --include-package my_library"
sys.argv = command.split()

if __name__ == '__main__':
    print(sys.argv)
    main()
