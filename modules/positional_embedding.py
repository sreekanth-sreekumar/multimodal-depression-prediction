import torch
import torch.nn as nn
import math

class PositonalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding_dim = embed_dim

    def forward(self, input):
        _, seq_len = input.size()
        max_pos = seq_len + 1
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_pos, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_pos, -1)
        return emb