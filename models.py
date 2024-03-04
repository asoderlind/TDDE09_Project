import torch
from torch import nn


class FixedWindowModel(nn.Module):
    def __init__(
        self,
        embedding_specs: list[tuple[int, int, int]],
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        # Create the embeddings based on the given specifications
        self.embeddings = nn.ModuleList()
        for n, num_embeddings, embedding_dim in embedding_specs:
            embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
            nn.init.normal_(embedding.weight, std=1e-2)
            for _ in range(n):
                self.embeddings.append(embedding)

        # Set up the FFN
        input_dim = sum(e.embedding_dim for e in self.embeddings)
        self.pipe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = [e(x[..., i]) for i, e in enumerate(self.embeddings)]
        return self.pipe(torch.cat(embedded, -1))
