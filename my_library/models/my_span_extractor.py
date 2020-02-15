"""
@Time    :2020/2/15 21:20
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
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple
from allennlp.nn import util

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@SpanExtractor.register("my_endpoint")
class MyEndpointSpanExtractor(EndpointSpanExtractor):
    def __init__(
            self,
            input_dim: int,
            combination: str = "x,y",
            num_width_embeddings: int = None,
            span_width_embedding_dim: int = None,
            bucket_widths: bool = False,
            use_exclusive_start_indices: bool = False,
    ) -> None:
        super(MyEndpointSpanExtractor, self).__init__(input_dim, combination, num_width_embeddings, span_width_embedding_dim, bucket_widths, use_exclusive_start_indices)
        self.linear = torch.nn.Linear(input_dim * len(combination.split(',')), input_dim)
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.LongTensor = None,
        span_indices_mask: torch.LongTensor = None,
    ):
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            if sequence_tensor.size(-1) != self._input_dim:
                raise ValueError(
                    f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                    f"received ({self._input_dim})."
                )
            start_embeddings = util.batched_index_select(sequence_tensor, span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP `SpanField` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).long().unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * (1 - start_sentinel_mask.squeeze(-1))

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(
                    f"Adjusted span indices must lie inside the the sequence tensor, "
                    f"but found: exclusive_span_starts: {exclusive_span_starts}."
                )

            start_embeddings = util.batched_index_select(sequence_tensor, exclusive_span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            float_start_sentinel_mask = start_sentinel_mask.float()
            start_embeddings = (
                    start_embeddings * (1 - float_start_sentinel_mask)
                    + float_start_sentinel_mask * self._start_sentinel
            )

        combined_tensors = util.combine_tensors(
            self._combination, [start_embeddings, end_embeddings]
        )

        combined_tensors = self.linear(combined_tensors)
        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(
                    span_ends - span_starts, num_total_buckets=self._num_width_embeddings
                )
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            combined_tensors = torch.cat([combined_tensors, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return combined_tensors * span_indices_mask.unsqueeze(-1).float()

        return combined_tensors
