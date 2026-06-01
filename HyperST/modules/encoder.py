import torch
from torch import nn
import math
from timm.models.vision_transformer import Mlp , Attention      

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x


class ResMLPBlock(nn.Module):

    def __init__(self, hidden_size, mlp_ratio=2.0, drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))
    
class ResMLPEncoder(nn.Module):

    def __init__(self, in_features, hidden_size, mlp_ratio=2.0, drop=0., depth=2):
        super().__init__()

        self.linear = nn.Linear(in_features, hidden_size)

        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_size, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)
        ])

        self.pre_norm = nn.LayerNorm(in_features)

        self.post_norm = nn.LayerNorm(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def forward(self, x):
        x = self.pre_norm(x)
        x = self.linear(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_norm(x)
        return x