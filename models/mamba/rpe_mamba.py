r"""Mamba with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


from models.transformer.output_layer import AttentionOutput

from einops import rearrange, repeat
from models.mamba.vanilla_mamba import Mamba


class RPEMamba(nn.Module):
    def __init__(self, d_model, dropout=None):
        super(RPEMamba, self).__init__()
        self.d_model = d_model
        self.mamba = Mamba(d_model)



    def forward(self, input_states, position_states):
        hidden_states = self.mamba(input_states)
        return hidden_states, input_states


class RPEMambaLayer(nn.Module):
    # d_model is hidden_dim
    # input_states (Tensor): (B, N, 3)
    # position_states (Tensor): (B, N, C)

    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPEMambaLayer, self).__init__()
        self.mamba = RPEMamba(d_model, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.mamba(
            input_states,
            position_states,
        )
        
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_states)
        output_states = self.output(hidden_states)
        return output_states, attention_scores



class PEMambaLayer(nn.Module):
    # d_model is hidden_dim
    # input_states (Tensor): (B, N, 3)
    # position_states (Tensor): (B, N, C)

    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPEMambaLayer, self).__init__()
        self.mamba = RPEMamba(d_model, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.mamba(
            input_states,
            position_states,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_states)
        output_states = self.output(hidden_states)
        return output_states, attention_scores
