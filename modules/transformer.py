import torch
from torch import nn
from modules.positional_embedding import PositonalEmbedding
import math

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor1, tesnsor2, device):
    dim1 = dim2 = tensor1.size()
    if tensor2 is not None:
        dim2 = tensor2.size()
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    future_mask.to(device)
    return future_mask[:dim1, :dim2]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout
        )

        self.attn_mask = attn_mask

        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.res_dropout = nn.Dropout(p=res_dropout)
        self.relu = nn.ReLU()

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x_k=None, x_v=None, src_key_padding_mask=None):

        x = self.layer_norm(x)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None

        if x_k in None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=src_key_padding_mask, attn_mask=mask)
        else:
            x_k = self.layer_norm(x_k)
            x_v = self.layer_norm(x_v)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, key_padding_mask=src_key_padding_mask, attn_mask=mask)
        x = self.res_dropout(x)
        x = residual + x

        residual = x
        x = self.layer_norm(x)
        x = self.relu(self.fc1(x))
        x = self.relu_dropout(x)
        x = self.fc2(x)
        x = self.res_dropout(x)
        x = residual + x
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, 
                relu_dropout=0.0, res_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = nn.Dropout(p = embed_dropout)      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)

        self.positional_embedding = PositonalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim, 
                num_heads = num_heads, 
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x_in, x_in_k = None, x_in_v = None, src_key_padding_mask = None):
        x = self.embed_scale * x_in
        x += self.positional_embedding(x_in.transpose(0,1)[:, :, 0].transpose(0,1))
        x = self.dropout(x)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            x_k += self.positional_embedding(x_in_k.transpose(0,1)[:, :, 0].transpose(0,1))
            x_v += self.positional_embedding(x_in_v.transpose(0,1)[:, :, 0].transpose(0,1))
            x_k = self.dropout(x_k)
            x_v = self.dropout(x_v)

        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v, src_key_padding_mask=src_key_padding_mask)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
            intermediates.append(x)

        x = self.layer_norm(x)

        return x