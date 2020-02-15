"""
@Time    :2020/2/14 23:10
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
import random
import os
import logging
import collections
from pathlib import Path

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import RegularizerApplicator
from allennlp.nn.util import clone, get_lengths_from_binary_sequence_mask
from allennlp.modules import Seq2SeqEncoder, MatrixAttention
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple, Optional

from my_library import defaults
from my_library.models.mymodel import MyModel, Encoder, Decoder
from my_library.models.sanyuanzuMetric import SanyuanzuMetric
from my_library.myutils import get_span_list, matrix_decode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@Model.register('onestagemodel')
class OneStageModel(MyModel):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 yingshe_encoder: Seq2SeqEncoder,
                 attention_decoder: MatrixAttention,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.encoder = encoder
        self.decoder_start_list = clone(attention_decoder, len(defaults.relation2id))
        self.decoder_end_list = clone(attention_decoder, len(defaults.relation2id))
        self.yingshe_encoder_list = clone(yingshe_encoder, 2)
        self.valid_metric = SanyuanzuMetric()
        self.flag = True
    def forward(self, tokens, target_start = None, target_end = None, metadata = None):

        output = {}

        encoded_dct = self.encoder(tokens)
        # shape: [batchsize, seq_len, encoded_size]
        encoded_info = encoded_dct['encoded_tensor']
        # shape: [batchsize, seq_len]
        mask = encoded_dct['mask']

        batchsize, seq_len, dim_size = encoded_info.shape

        encoded_info_1 = self.yingshe_encoder_list[0](encoded_info, mask)
        encoded_info_2 = self.yingshe_encoder_list[1](encoded_info, mask)
        decoded_start_list = torch.stack([decoder(encoded_info_1, encoded_info_1) for decoder in self.decoder_start_list], dim=-1)
        decoded_end_list = torch.stack([decoder(encoded_info_2, encoded_info_2) for decoder in self.decoder_end_list], dim=-1)
        assert decoded_start_list.shape == decoded_end_list.shape == (batchsize, seq_len, seq_len, len(defaults.relation2id))
        if self.training:
            output['loss'] = self.calculate_loss(decoded_start_list, target_start, mask) + self.calculate_loss(decoded_end_list, target_end, mask)
            return output
        # 预测阶段
        else:
            output = collections.defaultdict(list)
            assert mask.shape == (batchsize, seq_len)
            mask_length = get_lengths_from_binary_sequence_mask(mask)
            for decoded_start, decoded_end, length, m_data in zip(decoded_start_list, decoded_end_list, mask_length, metadata):
                pred_spo_list = []
                text = m_data['text']
                spo_list = m_data['spo_list']
                # 针对49种关系而言
                # shape: [relationship, seq_len, seq_len]
                decoded_start = decoded_start.permute(2, 0, 1).contiguous()
                decoded_end = decoded_end.permute(2, 0, 1).contiguous()
                assert decoded_start.shape == (len(defaults.relation2id), seq_len, seq_len)
                for i, (m1, m2) in enumerate(zip(decoded_start, decoded_end)):
                    predicate = defaults.id2relation[i]
                    s_o_spans = matrix_decode(m1, m2, length.item())
                    for s_span, o_span in s_o_spans:
                        s = text[s_span[0]: s_span[1] + 1]
                        o = text[o_span[0]: o_span[1] + 1]
                        pred_spo_list.append({"predicate": predicate, "object": o, "subject": s})
                if self.flag:
                    logger.info(pred_spo_list)
                    self.flag = False
                self.valid_metric(spo_list, pred_spo_list)
                output[text].append({'text':text, 'spo_list': pred_spo_list})
        self.flag = True
        return dict(output)

    def calculate_loss(self, pred, target, mask):
        # pred: shape:[batchsize, seq_len, seq_len, 49]
        assert len(pred.shape) == 4 and len(target.shape) == 4
        assert pred.shape == target.shape
        assert mask.shape == pred.shape[:2]
        batchsize, seq_len, _, r_num = pred.shape
        # shape: (batchsize, )
        mask_length = get_lengths_from_binary_sequence_mask(mask)
        # 设置新的mask矩阵
        new_mask = torch.zeros((batchsize, seq_len, seq_len, r_num)).to(mask.device)
        for i, v in enumerate(mask_length):
            v = v.item()
            new_mask[i, :v, :v, :] = 1
        count_all = mask_length.sum()
        youxiao = target.sum()
        assert count_all.item() >= youxiao.item()
        # 对一个batchsize的范围内，进行平均化
        # 这个是解决正负样本的方法

        new_mask[target==1] = (count_all - youxiao) / (youxiao + 1e-20)
        return binary_cross_entropy_with_logits(pred, target, new_mask, reduction = 'sum')





    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.training:
            return {}
        else:
            return self.valid_metric.get_metric(reset)


