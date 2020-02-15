"""
@Time    :2020/2/13 21:11
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
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
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

@Model.register('threestagemodel')
class ThreeStageModel(MyModel):
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 first_decoder: Seq2SeqEncoder,
                 second_decoder: Seq2SeqEncoder,
                 third_decoder: Seq2VecEncoder,
                 span_extractor:  SpanExtractor, # to determine
                 mode: str,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.encoder = encoder
        self.first_decoder_list = clone(first_decoder, 2)
        self.second_decoder_list = clone(second_decoder, 2)
        self.thrid_decoder = third_decoder # 仅作关系判别
        self.span_extractor_list = clone(span_extractor, 2)
        self.valid_metric = SanyuanzuMetric()
        assert mode in ['sop', 'osp']
        self.mode = mode

    def forward(self, tokens, one_span=None, two_span = None, one_s=None, one_e=None, two_s=None, two_e=None, relationship = None, metadata=None):
        output = {}
        encoded_dct = self.encoder(tokens)
        # shape: [batchsize, seq_len, encoded_size]
        encoded_info = encoded_dct['encoded_tensor']
        # shape: [batchsize, seq_len]
        mask = encoded_dct['mask']

        # shape: [batchsize, seq_len, 1]
        one_start_tensor = self.first_decoder_list[0](encoded_info)
        #shape: [batchsize, seq_len, 1]
        one_end_tensor = self.first_decoder_list[1](encoded_info)
        # 训练阶段
        if self.training:
            mask = mask.unsqueeze(-1).float()
            assert len(mask.shape) == 3
            self.calulate_one_loss(one_s, one_e, one_start_tensor, one_end_tensor, mask, output)

            predicate_second_s, predicate_second_e = self.get_second_predict(encoded_info, one_span, mask)
            self.calulate_second_loss(two_s, two_e, predicate_second_s, predicate_second_e, mask, output)

            pred_relationship = self.get_third_predict(encoded_info, one_span, two_span, mask)
            self.calulate_third_loss(relationship, pred_relationship, output)
            output['loss'] = output['subject_loss'] + output['object_loss'] + output['relationship_loss']
            return output
        #TODO()调试预测阶段代码
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
                    # 如果mode是s2po的话，则说明s是one
                    one_word: str = text[span[0]: span[1] + 1] # 解出了词

                    span = torch.tensor(span).view(1,1,-1).to(mask_t.device)# 确保GPU能正常训练
                    assert span.shape == (1, 1, 2)
                    predicate_second_s, predicate_second_e = self.get_second_predict(encode_info_t, span, mask_t)
                    # 压缩一下，0 ~ 1的范围
                    predicate_second_s = torch.sigmoid_(predicate_second_s)
                    predicate_second_e = torch.sigmoid_(predicate_second_e)
                    # shape: (batchsize, seq_len)
                    predicate_second_s = predicate_second_s.squeeze(-1).float() * mask_t.float()
                    predicate_second_e = predicate_second_e.squeeze(-1).float() * mask_t.float()
                    predicate_second_s = predicate_second_s.squeeze(0)
                    predicate_second_e = predicate_second_e.squeeze(0)
                    assert predicate_second_s.shape == (seq_len, ), f"{predicate_second_s.shape}"

                    two_span_list = get_span_list(predicate_second_s, predicate_second_e, defaults.yuzhi)
                    for two_span in two_span_list:
                        two_word = text[two_span[0]: two_span[1] + 1] # 解出了另一个词
                        two_span = torch.tensor(two_span).view(1,1,-1).to(mask_t.device)# 确保GPU能正常训练
                        # shape: [batchsize, 49]
                        predicate_relationship = self.get_third_predict(encode_info_t, span, two_span, mask_t)
                        assert predicate_relationship.shape == (1, len(defaults.relation2id)), f'{predicate_relationship.shape}'
                        predicate_relationship = torch.sigmoid(predicate_relationship).view(-1)
                        pred_idx = torch.where(predicate_relationship > defaults.yuzhi)[0]

                        for idx in pred_idx:
                            r = defaults.id2relation[idx.item()] # 真正名称
                            if self.mode == 'sop':
                                pred_spo_list.append({"predicate": r, "object": two_word, "subject": one_word})
                            else:
                                pred_spo_list.append({"predicate": r, "object": one_word, "subject": two_word})

                self.valid_metric(spo_list, pred_spo_list)
                output[text].append({'text':text, 'spo_list': pred_spo_list})
                return dict(output)

    def mix(self, encoded_info, embeddings: List[torch.Tensor]):
        "这里是求平均，应该可以有其他的融合方法，看论文需不需要"
        for tensor in embeddings:
            assert encoded_info.shape == tensor.shape
        mixed_indo = (encoded_info + sum(embeddings)) / (1 + len(embeddings))
        return mixed_indo

    def get_second_predict(self, encoded_info, span, mask):
        "将span范围内的向量想办法融合与原数据进行融合"
        batchsize, seq_len, dim_size = encoded_info.shape
        # shape: [batchsize, 1, 2]   参考(batch_size, num_spans, 2)
        span = span.view(batchsize, 1, 2)
        assert span.device == encoded_info.device
        assert span.shape == (batchsize, 1, 2), f"shape is {span.shape}"
        # shape: [batchsize, 1, dim_size]
        one_embedding = self.span_extractor_list[0](encoded_info, span, mask)
        # 这里可以相加，也可以其他操作，看论文需不需要其他操作
        # 重复到 shape: [batchsize, seq_len, dim_size]
        one_embedding = one_embedding.repeat(1, seq_len, 1)
        # shape: [batchsize, seq_len, dim_size]
        mixed_info = self.mix(encoded_info, [one_embedding, ])

        # shape: [batchsize, seq_len, 49]
        predicate_second_s = self.second_decoder_list[0](mixed_info, mask.squeeze(-1))
        predicate_second_e = self.second_decoder_list[1](mixed_info, mask.squeeze(-1))
        return predicate_second_s, predicate_second_e

    def get_third_predict(self, encoded_info, first_span, second_span, mask):
        "将span范围内的向量想办法融合与原数据进行融合"
        batchsize, seq_len, dim_size = encoded_info.shape
        # shape: [batchsize, 1, 2]   参考(batch_size, num_spans, 2)
        span = first_span.view(batchsize, 1, 2)
        assert span.device == encoded_info.device
        assert span.shape == (batchsize, 1, 2), f"shape is {span.shape}"
        # shape: [batchsize, 1, dim_size]
        one_embedding = self.span_extractor_list[0](encoded_info, span, mask) #抽取第一个，要用0号
        # 这里可以相加，也可以其他操作，看论文需不需要其他操作
        # 重复到 shape: [batchsize, seq_len, dim_size]
        one_embedding = one_embedding.repeat(1, seq_len, 1)

        span = second_span.view(batchsize, 1, 2)
        assert span.shape == (batchsize, 1, 2), f"shape is {span.shape}"
        assert span.device == encoded_info.device
        # shape: [batchsize, 1, dim_size]
        two_embedding = self.span_extractor_list[1](encoded_info, span, mask)
        # 这里可以相加，也可以其他操作，看论文需不需要其他操作
        # 重复到 shape: [batchsize, seq_len, dim_size]
        two_embedding = two_embedding.repeat(1, seq_len, 1)

        # shape: [batchsize, seq_len, dim_size]
        mixed_info = self.mix(encoded_info, [one_embedding, two_embedding])

        # 我们将subject，object和原句的信息融合了之后，就要进行关系判断了

        # shape: [batchsize, seq_len, 49]
        pred_relationship = self.thrid_decoder(mixed_info, mask.squeeze(-1))
        return pred_relationship


    def calulate_one_loss(self, one_s, one_e, one_start_tensor, one_end_tensor, mask, output):
        "例如，计算subject的loss值，这里集中处理了，保证forward里的逻辑会比较清晰"
        lossa = binary_cross_entropy_with_logits(one_start_tensor, one_s, mask)
        lossb = binary_cross_entropy_with_logits(one_end_tensor, one_e, mask)
        if self.mode == 's2po':
            output['subject_loss'] = lossa + lossb
        else:
            output['object_loss'] = lossa + lossb
        return output
    def calulate_second_loss(self, s, e, start_tensor, end_tensor, mask, output):
        "例如，计算object的loss值，这里集中处理了，保证forward里的逻辑会比较清晰"
        lossa = binary_cross_entropy_with_logits(start_tensor, s, mask)
        lossb = binary_cross_entropy_with_logits(end_tensor, e, mask)
        # 因为是在预测多的那个，故要反过来。
        if self.mode == 's2po':
            output['object_loss'] = lossa + lossb
        else:
            output['subject_loss'] = lossa + lossb
        return output
    def calulate_third_loss(self, r, pr, output):
        # shape: [batchsize, 49]
        assert r.shape == pr.shape
        assert len(r.shape) == 2
        loss = binary_cross_entropy_with_logits(pr, r)
        output['relationship_loss'] = loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.training:
            return {}
        else:
            return self.valid_metric.get_metric(reset)


