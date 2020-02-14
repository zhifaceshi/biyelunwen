"""
@Time    :2020/2/14 11:33
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
from typing import List, Dict, Tuple
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from pathlib import Path
from allennlp.commands import main
import fire
def run(exp_name, model_name):
    log_name = f'{exp_name}_{model_name}.log'
    workding = Path("/storage/gs2018/liangjiaxi/bishe/output/")
    expname = workding / exp_name
    model_dir = expname /  model_name
    OUTPUT_FILE = workding / 'mylogs' / log_name

    archive_file = model_dir / 'model.tar.gz'
    WEIGHTS_FILE = model_dir / 'best.th'
    CUDA_DEVICE = 0

    test_data = '/storage/gs2018/liangjiaxi/bishe/data/processed_data/_test.txt'

    command = f"allennlp evaluate  {archive_file}  {test_data}   --output-file {OUTPUT_FILE} --weights-file { WEIGHTS_FILE} --cuda-device {CUDA_DEVICE} --include-package my_library"
    sys.argv = command.split()
    print(sys.argv)
    main()
if __name__ == '__main__':
    fire.Fire(run)
