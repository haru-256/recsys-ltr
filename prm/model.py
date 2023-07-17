import torch
import torch.nn as nn


class SparseEmbedding(nn.Module):
    def __init__(self, cardinalities: dict[str, int], embedding_dim: int):
        """
        Embedding layer for sparse features.
        Embedding each categorical feature into same `embedding_dims` dim.
        Although each categorical feature is jagged, this means each sparse feature has different cardinality, we embedding those into same `embedding_dims` dim.

        Args:
            cardinalities: a dict of sparse feature name and its cardinality.
            embedding_dim: embedding dimension.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cardinalities = cardinalities

        self.category_embeddings = nn.ModuleDict(
            {
                feature_name: nn.EmbeddingBag(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    mode="mean",
                )
                for feature_name, num_embeddings in cardinalities.items()
            }
        )

    def forward(
        self,
        inputs: torch.LongTensor,
        feature_names: list[str],
    ) -> dict[str, torch.Tensor]:
        """forward process.

        Args:
            inputs: a category tensor of size: (`batch_size` x `num_features`).
            feature_names: a list of category feature names. The order of this should be same as inputs

        Returns:
            embedded features dict. This length is `num_features` and its element shape is (`batch_size` x `embedding_dims`)
        """
        outputs = {}
        for idx, feature_name in enumerate(feature_names):
            input_ = inputs[:, idx].reshape(-1, 1)
            output = self.category_embeddings[feature_name](input_)
            outputs[feature_name] = output

        return outputs
