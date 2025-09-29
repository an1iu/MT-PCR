import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.transformer.output_layer import AttentionOutput

from flash_attn import flash_attn_func

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, causal=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        if dropout is None or dropout <= 0:
            self.dropout_p = 0.0
        else:
            self.dropout_p = dropout

    def forward(self, input_q, input_k, input_v, **kwargs):
        """
        Args:
            input_q, input_k, input_v: (B, L, C)
        Returns:
            hidden_states: (B, L, C)
            attention_scores: None (FlashAttention 不显式返回 attention matrix)
        """
        B, N, C = input_q.size()
        q = rearrange(self.proj_q(input_q), 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b l (h d) -> b l h d', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b l (h d) -> b l h d', h=self.num_heads)

        q = q.half()
        k = k.half()
        v = v.half()
        # 使用 FlashAttention 进行高效注意力计算

        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p,
            softmax_scale=None,
            causal=self.causal,
            return_attn_probs=False
        )  # (B, L, H, D)
        out = out.float()
        out = rearrange(out, 'b l h d -> b l (h d)')
        return out, None  # 不再返回 attention_scores


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(AttentionLayer, self).__init__()
        self.attention = FlashMultiHeadAttention(d_model, num_heads, dropout)
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class FlashTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='relu'):
        super(FlashTransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_heads, dropout)
        self.output = AttentionOutput(d_model, dropout, activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
