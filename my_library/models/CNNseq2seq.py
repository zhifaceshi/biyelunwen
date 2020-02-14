"""
@Time    :2019/10/17 19:04
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json

from allennlp.modules import Seq2SeqEncoder
from tqdm import tqdm
import random
from pprint import pprint
import os
import collections
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


from typing import Optional, Tuple

from overrides import overrides
import torch
from torch.nn import Conv1d, Linear

from allennlp.nn import Activation


@Seq2SeqEncoder.register("cnn_seq2seq")
class CnnSeq2SeqEncoder(Seq2SeqEncoder):
    """
    A ``CnnEncoder`` is a combination of multiple convolution layers and max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),  # pylint: disable=bad-whitespace
                 ) -> None:
        super(CnnSeq2SeqEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes

        # 确认卷积核的大小是基数
        for ngram_size in self._ngram_filter_sizes:
            assert ngram_size % 2 == 1
        """
        torch.nn.Conv1d(last_dim, layer[1] * 2, layer[0],
                                           stride=1, padding=layer[0] - 1, bias=True)"""
        self._convolution_layers = [Conv1d(in_channels=self._embedding_dim,
                                           out_channels=self._num_filters,
                                           padding = ngram_size // 2,
                                           kernel_size=ngram_size)
                                    for ngram_size in self._ngram_filter_sizes]

        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        # 多少个卷积核 * 卷积类型数目
        self._output_dim = self._num_filters * len(self._ngram_filter_sizes)

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        batch_size, seq_len, embedding_dims = tokens.size()
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2) # torch.Size([4, 768, 49])
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))

            t = convolution_layer(tokens) # torch.Size([4, 1, 49])
            assert t.shape == (batch_size, self._output_dim, seq_len)
            filter_outputs.append(
                    t
            )

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        result = torch.sum(filter_outputs) if len(filter_outputs) > 1 else filter_outputs[0]
        result /= len(filter_outputs)
        result = torch.transpose(result, 1, 2) # torch.Size([4, 49, 1])
        assert result.shape == (batch_size, seq_len, self._output_dim)
        return result.contiguous()
