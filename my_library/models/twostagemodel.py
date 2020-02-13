"""
@Time    :2020/2/13 9:44
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
from allennlp.nn.util import clone
from allennlp.modules import Seq2SeqEncoder
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple, Optional

from my_library import defaults
from my_library.models.mymodel import MyModel, Encoder, Decoder
from my_library.models.sanyuanzuMetric import SanyuanzuMetric
from my_library.myutils import get_span_list

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@Model.register('twostagemodel')
class TwoStageModel(MyModel):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 one_decoder: Seq2SeqEncoder,
                 many_decoder: Seq2SeqEncoder,
                 span_extractor:  SpanExtractor, # to determine
                 mode: str,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.encoder = encoder
        self.one_decoder_list = clone(one_decoder, 2)
        self.many_decoder_list = clone(many_decoder, 2)
        self.span_extractor = span_extractor
        self.valid_metric = SanyuanzuMetric()
        assert mode in ['s2po', 'o2ps']
        self.mode = mode
    def forward(self, tokens, span=None, one_s=None, one_e=None, many_s=None, many_e=None, metadata=None):

        output = {}

        encoded_dct = self.encoder(tokens)
        # shape: [batchsize, seq_len, encoded_size]
        encoded_info = encoded_dct['encoded_tensor']
        # shape: [batchsize, seq_len]
        mask = encoded_dct['mask']

        # shape: [batchsize, seq_len, 1]
        one_start_tensor = self.one_decoder_list[0](encoded_info)
        #shape: [batchsize, seq_len, 1]
        one_end_tensor = self.one_decoder_list[1](encoded_info)
        # 训练阶段
        if self.training:
            mask = mask.unsqueeze(-1).float()
            self.calulate_one_loss(one_s, one_e, one_start_tensor, one_end_tensor, mask, output)
            # 这一步在预测阶段也可以复用，故抽取出，写成了函数

            predicate_many_s, predicate_many_e = self.get_many_predict(encoded_info, span, mask)
            self.calulate_many_loss(many_s, many_e, predicate_many_s, predicate_many_e, mask, output)
            output['loss'] = output['subject_loss'] + output['object_loss']
            return output
        # 预测阶段
        else:
            output = collections.defaultdict(list)

            batchsize, seq_len, dim_size = encoded_info.shape

            # 将每个元素，压缩到0 ~ 1之间
            # shape: [batchsize, seq_len, dim_size]
            one_start_tensor = torch.sigmoid_(one_start_tensor)
            one_end_tensor = torch.sigmoid_(one_end_tensor)
            # shape: [batchsize, seq_len, 1]
            one_start_tensor = one_start_tensor * mask.unsqueeze(-1).float()
            one_end_tensor = one_end_tensor * mask.unsqueeze(-1).float()
            assert one_start_tensor.shape == (batchsize, seq_len, 1)

            # shape: [batchsize, seq_len]
            one_start_tensor = one_start_tensor.squeeze(-1).float()
            one_end_tensor = one_end_tensor.squeeze(-1).float()
            assert one_start_tensor.shape == (batchsize, seq_len)

            assert one_start_tensor.size(0) == one_end_tensor.size(0) == encoded_info.size(0) == len(metadata) == mask.size(0)
            for one_start_t, one_end_t, encode_info_t, spo_list_t, mask_t in zip(one_start_tensor, one_end_tensor, encoded_info, metadata, mask):
                # shape: [1, seq_len]
                mask_t = mask_t.unsqueeze(0)
                assert mask_t.shape == (1, seq_len)

                # shape: [1, seq_len, dim_size]
                encode_info_t = encode_info_t.unsqueeze(0)
                one_span_lst = get_span_list(one_start_t, one_end_t, defaults.yuzhi)
                text = spo_list_t['text']
                spo_list = spo_list_t['spo_list']
                pred_spo_list = []

                for span in one_span_lst:
                    # 如果mode是s2po的话，则说明s是one，object是many
                    one_word: str = text[span[0]: span[1] + 1]
                    span = torch.tensor(span).view(1,1,-1)
                    predicate_many_s, predicate_many_e = self.get_many_predict(encode_info_t, span, mask_t)
                    # shape: [1, seq_len, 49]
                    predicate_many_s = torch.sigmoid_(predicate_many_s)
                    predicate_many_e = torch.sigmoid_(predicate_many_e)
                    # shape: [1, 49, seq_len]
                    predicate_many_s = predicate_many_s.permute(0, 2, 1).squeeze(0)
                    predicate_many_e = predicate_many_e.permute(0, 2, 1).squeeze(0)
                    assert predicate_many_s.shape == (49, seq_len), f"{predicate_many_s.shape}"
                    # 遍历49种关系，解出相应的答案
                    for relationship_id, (ps, pe) in enumerate(zip(predicate_many_s, predicate_many_e)):
                        many_span_lst = get_span_list(ps, pe, defaults.yuzhi)
                        predicate = defaults.id2relation[relationship_id]
                        for i, j in many_span_lst:
                            many_one_word: str = text[i: j+1]
                            if self.mode == 's2po':
                                pred_spo_list.append({"predicate": predicate, "object": many_one_word, "subject": one_word})
                            else:
                                pred_spo_list.append({"predicate": predicate, "object": one_word, "subject": many_one_word})

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


