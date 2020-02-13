"""
@Time    :2020/2/12 23:12
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
import random
import os
import logging
import collections
from pathlib import Path

import numpy
import torch
from allennlp.common import FromParams, Registrable
from allennlp.data import Instance, Vocabulary, TextFieldTensors
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import util, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MyModel(Model):
    @overrides
    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        我省略了复杂繁琐的检查，因为这会导致模型最后可能没有输出
        :param instances:
        :return:
        """
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))
            return outputs
            # instance_separated_output: List[Dict[str, numpy.ndarray]] = [
            #     {} for _ in dataset.instances
            # ]
            # for name, output in list(outputs.items()):
            #     if isinstance(output, torch.Tensor):
            #         # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
            #         # This occurs with batch size 1, because we still want to include the loss in that case.
            #         if output.dim() == 0:
            #             output = output.unsqueeze(0)
            #
            #         if output.size(0) != batch_size:
            #             self._maybe_warn_for_unseparable_batches(name)
            #             continue
            #         output = output.detach().cpu().numpy()
            #     elif len(output) != batch_size:
            #         self._maybe_warn_for_unseparable_batches(name)
            #         continue
            #     for instance_output, batch_element in zip(instance_separated_output, output):
            #         instance_output[name] = batch_element
            # return instance_separated_output

@Seq2SeqEncoder.register("encoder")
class Encoder(torch.nn.Module, Registrable):
    "将输入的文本进行编码，编码成等长的向量"
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder = None,
                ) -> None:
        super().__init__()
        self.embedder = text_field_embedder
        self.encoder = encoder

    def forward(self, text: TextFieldTensors) -> Dict[str, torch.Tensor]:

        # shape: [batchsize, seq_len, dim_size]
        embedding = self.embedder(text)
        mask = get_text_field_mask(text)
        if self.encoder is not None:
            # shape: [batchsize, seq_len, encoder_output_dim]
            embedding = self.encoder(embedding, mask)
        return {
            'encoded_tensor': embedding,
            'mask': mask
        }
@Seq2SeqEncoder.register("decoder")
class Decoder(torch.nn.Module, Registrable):
    "将向量进行输出，输出成等长的向量"
    def __init__(self, vocab: Vocabulary,
                 decoder: Seq2SeqEncoder,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, encoded_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        # shape: [batchsize, seq_len, embedding_size]
        decoder_tensor = self.decoder(encoded_tensor)
        return {
            'decoded_tensor': decoder_tensor
        }



