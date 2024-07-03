# test_sentiment_analysis.py

import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        embedding_dim: int = 16
        self.embed = nn.Embedding(vocabulary_size, embedding_dim)
        self.end_linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        out = self.embed(x)
        out = torch.mean(out, axis=1)
        out = self.end_linear(out)
        out = self.sigmoid(out)
        return out.round(decimals=4)
