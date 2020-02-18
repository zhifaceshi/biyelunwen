from overrides import overrides

import torch
import torch.nn

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("end_pooler")
class EndPooler(Seq2VecEncoder):
    """
    Just takes the first vector from a list of vectors (which in a transformer is typically the
    [CLS] token) and returns it.

    # Parameters

    embedding_dim: int, optional
        This isn't needed for any computation that we do, but we sometimes rely on `get_input_dim`
        and `get_output_dim` to check parameter settings, or to instantiate final linear layers.  In
        order to give the right values there, we need to know the embedding dimension.  If you're
        using this with a transformer from the `transformers` library, this can often be found with
        `model.config.hidden_size`, if you're not sure.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self._embedding_dim = input_dim
    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        # tokens is assumed to have shape (batch_size, sequence_length, embedding_dim).  We just
        # want the first token for each instance in the batch.
        assert len(tokens.shape) == 3
        mask_length = mask.sum(1).long() - 1
        batch_size, _, encoder_output_dim = tokens.size()
        expanded_indices = mask_length.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
        # Shape: (batch_size, 1, encoder_output_dim)
        final_encoder_output = tokens.gather(1, expanded_indices)
        final_encoder_output = final_encoder_output.squeeze(1)
        return final_encoder_output
