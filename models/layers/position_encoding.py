import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal

class PositionalEncoding(nn.Module):
    """Base class for positional encodings"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SinusoidalPositionalEncoding(PositionalEncoding):
    """Sinusoidal positional encoding for temporal sequence"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        original_d_model = d_model

        if d_model % 2 != 0:
            d_model += 1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if d_model != original_d_model:
            pe = pe[:, :original_d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, node_size, feature_size]
        Returns:
            Tensor of same shape with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0).unsqueeze(2)

class LearnablePositionalEncoding(PositionalEncoding):
    """Learnable positional encoding for temporal sequence"""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, node_size, feature_size]
        Returns:
            Tensor of same shape with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.embedding[:seq_len].unsqueeze(0).unsqueeze(2)

class LaplacianPositionalEncoding(PositionalEncoding):
    """Laplacian positional encoding for graph nodes"""
    def __init__(self, num_nodes: int, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, node_size, feature_size]
        Returns:
            Tensor of same shape with positional encoding added
        """
        return x + self.embedding.unsqueeze(0).unsqueeze(1)

class CombinedPositionalEncoding(nn.Module):
    """Combines temporal and graph positional encodings"""
    def __init__(
        self,
        feature_size: int,
        max_seq_len: int,
        num_nodes: int,
        temporal_encoding: Literal['sinusoidal', 'learnable'] = 'sinusoidal',
        node_encoding: Literal['laplacian'] = 'laplacian'
    ):
        super().__init__()
        
        # Initialize temporal positional encoding
        if temporal_encoding == 'sinusoidal':
            self.temporal_encoding = SinusoidalPositionalEncoding(feature_size, max_seq_len)
        elif temporal_encoding == 'learnable':
            self.temporal_encoding = LearnablePositionalEncoding(max_seq_len, feature_size)
        else:
            raise ValueError(f"Unsupported temporal encoding type: {temporal_encoding}")
            
        # Initialize node positional encoding
        if node_encoding == 'laplacian':
            self.node_encoding = LaplacianPositionalEncoding(num_nodes, feature_size)
        else:
            raise ValueError(f"Unsupported node encoding type: {node_encoding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply both temporal and node positional encodings
        
        Args:
            x: Tensor of shape [batch_size, seq_len, node_size, feature_size]
        Returns:
            Tensor of same shape with both positional encodings added
        """
        # Apply temporal encoding
        x = self.temporal_encoding(x)
        # Apply node encoding
        x = self.node_encoding(x)
        return x
