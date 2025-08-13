import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal


class MultiHeadAttention(nn.Module):
    """Base class for multi-head attention mechanisms"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_module = nn.Dropout(dropout)


    def _reshape_for_scores(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape tensor for attention computation"""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generic attention forward pass

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask

        Returns:
            tuple: (attention output, attention weights)
        """
        raise NotImplementedError


class SeparateAttention(MultiHeadAttention):
    """Implements separate temporal and spatial attention"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.spatial_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,  # [batch_size, seq_len, node_size, embed_dim]
        key: torch.Tensor,
        value: torch.Tensor,
        adj: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = query.size(0)
        seq_len = query.size(1)
        node_size = query.size(2)

        # Project inputs
        q = self.q_proj(query)  # [batch_size, seq_len, node_size, embed_dim]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Temporal attention
        # Reshape to [batch_size * node_size, seq_len, embed_dim]
        # q_temporal = q.transpose(1, 2).reshape(batch_size * node_size, seq_len, self.embed_dim)
        # k_temporal = k.transpose(1, 2).reshape(batch_size * node_size, seq_len, self.embed_dim)
        # v_temporal = v.transpose(1, 2).reshape(batch_size * node_size, seq_len, self.embed_dim)

        q_temporal = q.transpose(1, 2).reshape(batch_size * node_size, query.size(1), self.embed_dim)
        k_temporal = k.transpose(1, 2).reshape(batch_size * node_size, key.size(1), self.embed_dim)
        v_temporal = v.transpose(1, 2).reshape(batch_size * node_size, value.size(1), self.embed_dim)

        # Multi-head reshape
        q_t = self._reshape_for_scores(q_temporal, batch_size * node_size)
        k_t = self._reshape_for_scores(k_temporal, batch_size * node_size)
        v_t = self._reshape_for_scores(v_temporal, batch_size * node_size)

        # Compute temporal attention
        attn_weights_temporal = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scaling
        if attn_mask is not None:
            attn_weights_temporal = attn_weights_temporal.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights_temporal = F.softmax(attn_weights_temporal, dim=-1)
        attn_weights_temporal = self.dropout_module(attn_weights_temporal)
        temporal_out = torch.matmul(attn_weights_temporal, v_t)

        # Reshape temporal output
        temporal_out = temporal_out.transpose(1, 2).contiguous()
        temporal_out = temporal_out.view(batch_size, node_size, seq_len, self.embed_dim)
        temporal_out = temporal_out.transpose(1, 2)

        # Spatial attention
        # Reshape to [batch_size * seq_len, node_size, embed_dim]
        # q_spatial = q.reshape(batch_size * seq_len, node_size, self.embed_dim)
        # k_spatial = k.reshape(batch_size * seq_len, node_size, self.embed_dim)
        # v_spatial = v.reshape(batch_size * seq_len, node_size, self.embed_dim)

        q_spatial = q.reshape(batch_size * query.size(1), node_size, self.embed_dim)
        k_spatial = k.reshape(batch_size * key.size(1), node_size, self.embed_dim)
        v_spatial = v.reshape(batch_size * value.size(1), node_size, self.embed_dim)

        # Multi-head reshape
        q_s = self._reshape_for_scores(q_spatial, batch_size * seq_len)
        k_s = self._reshape_for_scores(k_spatial, batch_size * seq_len)
        v_s = self._reshape_for_scores(v_spatial, batch_size * seq_len)

        # Compute spatial attention
        attn_weights_spatial = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scaling

        # add adj to contral the correlation
        adj_expanded = adj[0].unsqueeze(0).unsqueeze(0).expand_as(attn_weights_spatial)
        attn_weights_spatial = attn_weights_spatial * adj_expanded
        # get a inf_mask for adj which set 0 to -inf
        inf_mask = torch.zeros_like(attn_weights_spatial)
        inf_mask = inf_mask.masked_fill(adj_expanded == 0, -1e9)  # 或 -inf(traffic), -1e9(purchase, air), 具体值需要根据你的场景调整
        attn_weights_spatial += inf_mask

        # normalize weight
        attn_weights_spatial = F.softmax(attn_weights_spatial, dim=-1)
        attn_weights_spatial = self.dropout_module(attn_weights_spatial)
        spatial_out = torch.matmul(attn_weights_spatial, v_s)

        # Reshape spatial output
        spatial_out = spatial_out.transpose(1, 2).contiguous()
        spatial_out = spatial_out.view(batch_size, seq_len, node_size, self.embed_dim)

        # Combine temporal and spatial attention
        temporal_out = F.relu(self.temporal_proj(temporal_out))
        spatial_out  = F.relu(self.spatial_proj(spatial_out))

        output = self.out_proj(temporal_out + spatial_out)

        return output, (attn_weights_temporal, attn_weights_spatial)


class UnifiedAttention(MultiHeadAttention):
    """Implements unified attention across both temporal and spatial dimensions"""
    def forward(
        self,
        query: torch.Tensor,  # [batch_size, tgt_seq_len, node_size, embed_dim]
        key: torch.Tensor,    # [batch_size, src_seq_len, node_size, embed_dim]
        value: torch.Tensor,  # [batch_size, src_seq_len, node_size, embed_dim]
        adj: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        tgt_seq_len = query.size(1)
        src_seq_len = key.size(1)
        node_size = query.size(2)

        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to treat seq_len and node_size as a single dimension
        # [batch_size, seq_len * node_size, embed_dim]
        q = q.reshape(batch_size, tgt_seq_len * node_size, self.embed_dim)
        k = k.reshape(batch_size, src_seq_len * node_size, self.embed_dim)
        v = v.reshape(batch_size, src_seq_len * node_size, self.embed_dim)

        # Multi-head reshape
        q = self._reshape_for_scores(q, batch_size)
        k = self._reshape_for_scores(k, batch_size)
        v = self._reshape_for_scores(v, batch_size)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Update attention mask if needed
        if attn_mask is not None:
            # 需要调整 attn_mask 以匹配新的注意力矩阵维度
            # attn_mask 应该考虑到 node_size 的重复
            attn_mask = attn_mask.repeat_interleave(node_size, dim=-1).repeat_interleave(node_size, dim=-2)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_module(attn_weights)
        attention = torch.matmul(attn_weights, v)

        # Reshape output
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, tgt_seq_len, node_size, self.embed_dim)
        output = self.out_proj(attention)

        return output, attn_weights


class NaiveTemporalAttention(MultiHeadAttention):
    """Implements naive temporal attention with only temporal dimension"""
    def forward(
        self,
        query: torch.Tensor,  # [batch_size, tgt_seq_len, node_size, embed_dim]
        key: torch.Tensor,    # [batch_size, src_seq_len, node_size, embed_dim]
        value: torch.Tensor,  # [batch_size, src_seq_len, node_size, embed_dim]
        adj: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        tgt_seq_len = query.size(1)
        src_seq_len = key.size(1)
        node_size = query.size(2)

        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to treat seq_len and batch_size as a single dimension
        # [batch_size * node_size, seq_len, embed_dim]
        q = q.transpose(1, 2).reshape(batch_size * node_size, tgt_seq_len, self.embed_dim)
        k = k.transpose(1, 2).reshape(batch_size * node_size, src_seq_len, self.embed_dim)
        v = v.transpose(1, 2).reshape(batch_size * node_size, src_seq_len, self.embed_dim)

        # Multi-head reshape
        q = self._reshape_for_scores(q, batch_size * node_size)
        k = self._reshape_for_scores(k, batch_size * node_size)
        v = self._reshape_for_scores(v, batch_size * node_size)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_module(attn_weights)
        attention = torch.matmul(attn_weights, v)

        # Reshape output
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, node_size, tgt_seq_len, self.embed_dim).transpose(1, 2)
        output = self.out_proj(attention)

        return output, attn_weights


class AttentionFactory:
    """Factory class for creating attention modules"""
    @staticmethod
    def create_attention(
        attention_type: Literal['separate', 'unified', 'temporal'],
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ) -> MultiHeadAttention:
        if attention_type == 'separate':
            return SeparateAttention(embed_dim, num_heads, dropout, bias)
        elif attention_type == 'unified':
            return UnifiedAttention(embed_dim, num_heads, dropout, bias)
        elif attention_type == 'temporal':
            return NaiveTemporalAttention(embed_dim, num_heads, dropout, bias)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
