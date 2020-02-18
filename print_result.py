"""
@Time    :2020/2/17 9:24
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
import re
from pathlib import Path
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run(exp_name):
    root_pth = Path('/storage/gs2018/liangjiaxi/bishe/output')
    exp_dir = root_pth / exp_name
    for dir_name in exp_dir.glob('*'):
        file_name = "metrics.json"
        try:

            with open(exp_dir / dir_name / file_name) as f:
                data = json.load(f)
            p = data['best_validation_precision'] *100
            r = data['best_validation_recall']*100
            f1 = data['best_validation_f1']*100
            t_p = data['test_precision']*100
            t_r = data['test_recall']*100
            t_f1 = data['test_f1']*100
            print(f"{dir_name.stem} {p:.2f} {r:.2f} {f1:.2f} {t_p:.2f} {t_r:.2f} {t_f1:.2f}")
        except:
            pass

if __name__ == '__main__':
    run('exp2')



