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
                 attention_decoder: MatrixAttention,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.encoder = encoder
        self.decoder_start_list = clone(attention_decoder, len(defaults.relation2id))
        self.decoder_end_list = clone(attention_decoder, len(defaults.relation2id))
        self.valid_metric = SanyuanzuMetric()
    def forward(self, tokens, target_start, target_end, metadata):

        output = {}

        encoded_dct = self.encoder(tokens)
        # shape: [batchsize, seq_len, encoded_size]
        encoded_info = encoded_dct['encoded_tensor']
        # shape: [batchsize, seq_len]
        mask = encoded_dct['mask']

        batchsize, seq_len, dim_size = encoded_info.shape
        # 这里将padding的部分设置为0，方便计算
        encoded_info[mask == 0] = 0
        decoded_start_list = torch.stack([decoder(encoded_info) for decoder in self.decoder_start_list], dim=-1)
        decoded_end_list = torch.stack([decoder(encoded_info) for decoder in self.decoder_end_list], dim=-1)
        assert decoded_start_list.shape == decoded_end_list.shape == (batchsize, seq_len, seq_len, len(defaults.relation2id))
        if self.training:
            output['loss'] = binary_cross_entropy_with_logits(decoded_start_list, target_start) + binary_cross_entropy_with_logits(decoded_end_list, target_end)
            return output
        # 预测阶段
        else:
            output = collections.defaultdict(list)
            assert mask.shape == (batchsize, seq_len)
            mask_length = get_lengths_from_binary_sequence_mask(mask)
            for decoded_start, decoded_end, length, m_data in zip(decoded_start_list, decoded_end_list, mask_length, metadata['text']):
                pred_spo_list = []
                text = m_data['text']
                spo_list = m_data['spo_list']
                for i, m1, m2 in zip(decoded_start.split(-1), decoded_end.split(-1)):
                    predicate = defaults.id2relation[i]
                    s_o_spans = matrix_decode(m1, m2, length.item())

                    for s_span, o_span in s_o_spans:
                        s = text[s_span[0]: s_span[1] + 1]
                        o = text[o_span[0]: o_span[1] + 1]
                        pred_spo_list.append({"predicate": predicate, "object": o, "subject": s})
                self.valid_metric(spo_list, pred_spo_list)
                output[text].append({'text':text, 'spo_list': pred_spo_list})
        return dict(output)

    def mix(self, encoded_info, one_embedding):
        "这里是求平均，应该可以有其他的融合方法，看论文需不需要"
        assert encoded_info.shape == one_embedding.shape
        mixed_indo = (encoded_info + one_embedding) / 2
        return mixed_indo
    def get_many_predict(self, encoded_info, span, mask):
        "将span范围内的向量想办法融合与原数据进行融合"
        batchsize, seq_len, dim_size = encoded_info.shape
        # shape: [batchsize, 1, 2]   参考(batch_size, num_spans, 2)
        span = span.view(batchsize, 1, 2)
        assert span.shape == (batchsize, 1, 2), f"shape is {span.shape}"
        assert span.device == encoded_info.device == mask.device, f"{span.device} {encoded_info.device} {mask.device}"
        # shape: [batchsize, 1, dim_size]
        one_embedding = self.span_extractor(encoded_info, span, mask)
        # 这里可以相加，也可以其他操作，看论文需不需要其他操作
        # 重复到 shape: [batchsize, seq_len, dim_size]
        one_embedding = one_embedding.repeat(1, seq_len, 1)
        # shape: [batchsize, seq_len, dim_size]
        mixed_info = self.mix(encoded_info, one_embedding)

        # shape: [batchsize, seq_len, 49]
        predicate_many_s = self.many_decoder_list[0](mixed_info, mask.squeeze(-1))
        predicate_many_e = self.many_decoder_list[1](mixed_info, mask.squeeze(-1))
        return predicate_many_s, predicate_many_e


    def calulate_one_loss(self, one_s, one_e, one_start_tensor, one_end_tensor, mask, output):
        "例如，计算subject的loss值，这里集中处理了，保证forward里的逻辑会比较清晰"
        lossa = binary_cross_entropy_with_logits(one_start_tensor, one_s, mask)
        lossb = binary_cross_entropy_with_logits(one_end_tensor, one_e, mask)
        if self.mode == 's2po':
            output['subject_loss'] = lossa + lossb
        else:
            output['object_loss'] = lossa + lossb
        return output
    def calulate_many_loss(self, s, e, start_tensor, end_tensor, mask, output):
        "例如，计算object的loss值，这里集中处理了，保证forward里的逻辑会比较清晰"
        lossa = binary_cross_entropy_with_logits(start_tensor, s, mask)
        lossb = binary_cross_entropy_with_logits(end_tensor, e, mask)
        # 因为是在预测多的那个，故要反过来。
        if self.mode == 's2po':
            output['object_loss'] = lossa + lossb
        else:
            output['subject_loss'] = lossa + lossb
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.training:
            return {}
        else:
            return self.valid_metric.get_metric(reset)


