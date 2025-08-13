from typing import Optional, Literal

import torch
import torch.nn as nn

from .position_encoding import CombinedPositionalEncoding
from .attention import AttentionFactory


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer implementation"""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_type: Literal['separate', 'unified'] = 'separate',
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.attention = AttentionFactory.create_attention(
            attention_type=attention_type,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(
        self,
        t: torch.Tensor,
        src: torch.Tensor,
        adj: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, node_size, d_model]
            src_mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)

        src_ss = modulate(src, shift_msa, scale_msa)
        src_msa, _ = self.attention(
            src_ss, src_ss, src_ss, adj,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src_msa = gate_msa * src_msa

        src = src + self.dropout1(src_msa)
        if src.size(-1) > 1:
            src = self.norm1(src)

        src_ss = modulate(src, shift_mlp, scale_mlp)
        src_mlp = self.linear2(self.dropout(self.activation(self.linear1(src_ss))))
        src_mlp = gate_mlp * src_mlp
        src = src + self.dropout2(src_mlp)
        if src.size(-1) > 1:
            src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer implementation"""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_type: Literal['separate', 'unified'] = 'separate',
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.self_attention = AttentionFactory.create_attention(
            attention_type=attention_type,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        self.cross_attention = AttentionFactory.create_attention(
            attention_type=attention_type,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True)
        )

    def forward(
        self,
        t: torch.Tensor,
        tgt: torch.Tensor,
        adj: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, seq_len, node_size, d_model]
            memory: Optional encoder output for encoder-decoder attention
            tgt_mask: Optional self-attention mask
            memory_mask: Optional cross-attention mask
            tgt_key_padding_mask: Optional target key padding mask
            memory_key_padding_mask: Optional memory key padding mask
        """
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=-1)
        # scale + shift
        tgt_ss = modulate(tgt, shift_msa, scale_msa)
        tgt_msa, _ = self.self_attention(
            tgt_ss, tgt_ss, tgt_ss, adj,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        # scale
        tgt_msa = gate_msa * tgt_msa
        tgt = tgt + self.dropout1(tgt_msa)
        if tgt.size(-1) > 1:
            tgt = self.norm1(tgt)

        if memory is not None:
            # scale + shift
            tgt_ss = modulate(tgt, shift_mca, scale_mca)
            tgt_mca, _ = self.cross_attention(
                tgt_ss, memory, memory, 
                adj=adj,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            # scale
            tgt_mca = gate_mca * tgt_mca
            tgt = tgt + self.dropout2(tgt_mca)
            if tgt.size(-1) > 1:
                tgt = self.norm2(tgt)

        # scale + shift
        tgt_ss = modulate(tgt, shift_mlp, scale_mlp)
        tgt_mlp = self.linear2(self.dropout(self.activation(self.linear1(tgt_ss))))
        # scale
        tgt_mlp = gate_mlp * tgt_mlp
        tgt = tgt + self.dropout3(tgt_mlp)
        if tgt.size(-1) > 1:
            tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    """Full encoder stack"""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_type: Literal['separate', 'unified'] = 'separate',
        activation: nn.Module = nn.ReLU(),
        pos_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.pos_encoder = pos_encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_type=attention_type,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        t: torch.Tensor,
        src: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = src
        for layer in self.layers:
            output = layer(t, output, adj, mask, src_key_padding_mask)

        if output.size(-1) > 1:
            return self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Full decoder stack"""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_type: Literal['separate', 'unified'] = 'separate',
        activation: nn.Module = nn.ReLU(),
        pos_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.pos_encoder = pos_encoder
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_type=attention_type,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        t: torch.Tensor,
        tgt: torch.Tensor,
        adj: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.pos_encoder is not None:
            tgt = self.pos_encoder(tgt)

        output = tgt
        for layer in self.layers:
            output = layer(
                t,
                output,
                adj,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask
            )

        if output.size(-1) > 1:
            return self.norm(output)
        return output


class SpecialTokens:
    """Manages special tokens for the model"""
    def __init__(
        self,
        node_size: int,
        feature_size: int,
        init_type: Literal['zeros', 'random', 'onehot'] = 'random'
    ):
        self.node_size = node_size
        self.feature_size = feature_size

        if init_type == 'zeros':
            self.start_token = nn.Parameter(torch.zeros(node_size, feature_size))
            self.end_token = nn.Parameter(torch.zeros(node_size, feature_size))
        elif init_type == 'random':
            self.start_token = nn.Parameter(torch.randn(node_size, feature_size))
            self.end_token = nn.Parameter(torch.randn(node_size, feature_size))
        elif init_type == 'onehot':
            start = torch.zeros(node_size, feature_size)
            end = torch.zeros(node_size, feature_size)
            start[:, 0] = 1.0  # First position for START
            end[:, 1] = 1.0    # Second position for END
            self.start_token = nn.Parameter(start)
            self.end_token = nn.Parameter(end)
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")


class STTransformer(nn.Module):
    """Main transformer model with support for encoder-decoder and decoder-only modes"""
    def __init__(
        self,
        dim_feature: int,
        dim_node: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_seq_len: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        encoder_attention_type: Literal['separate', 'unified'] = 'separate',
        decoder_attention_type: Literal['separate', 'unified'] = 'separate',
        mode: Literal['encoder_decoder', 'decoder_only'] = 'encoder_decoder',
        temporal_pos_embedding: Literal['sinusoidal', 'learnable'] = 'sinusoidal',
        node_pos_embedding: Literal['laplacian'] = 'laplacian',
        special_tokens_init: Literal['zeros', 'random', 'onehot'] = 'random',
    ):
        super().__init__()
        self.d_model = dim_feature
        self.mode = mode

        # Initialize position encoding
        self.pos_encoder = CombinedPositionalEncoding(
            feature_size=dim_feature,
            max_seq_len=max_seq_len,
            num_nodes=dim_node,
            temporal_encoding=temporal_pos_embedding,
            node_encoding=node_pos_embedding
        )

        # Initialize special tokens
        self.special_tokens = SpecialTokens(
            node_size=dim_node,
            feature_size=dim_feature,
            init_type=special_tokens_init
        )

        # Initialize transformer components based on mode
        if mode == 'encoder_decoder':
            self.encoder = TransformerEncoder(
                d_model=dim_feature,
                nhead=num_heads,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_type=encoder_attention_type,
                pos_encoder=self.pos_encoder,
                activation=nn.SiLU(),
            )

        self.decoder = TransformerDecoder(
            d_model=dim_feature,
            nhead=num_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_type=decoder_attention_type,
            pos_encoder=self.pos_encoder,
            activation=nn.SiLU(),
        )

        self.output_projection = nn.Linear(dim_feature, dim_feature)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask == 0

    def forward(
        self,
        t: torch.Tensor,
        src: torch.Tensor,
        adj: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass supporting both encoder-decoder and decoder-only modes

        Args:
            src: Input sequence [batch_size, seq_len, node_size, d_model]
            tgt: Optional target sequence for training
            *_mask: Optional attention masks
            *_key_padding_mask: Optional key padding masks

        Returns:
            Output sequence [batch_size, seq_len, node_size, d_model]
        """
        if self.mode == 'encoder_decoder':
            # Standard encoder-decoder forward pass
            memory = self.encoder(t, src, adj, src_mask, src_key_padding_mask)
            if tgt.size(1) != t.size(1):
                t = torch.cat([t[:, [0], :, :], t, t[:, [-1], :, :]], dim=1)
            output = self.decoder(
                t,
                tgt,
                adj,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask
            )
        else:  # decoder_only mode
            # In decoder-only mode, we only use the decoder
            # and src is treated as the previous sequence
            if tgt is None:
                tgt = src
            if tgt.size(1) != t.size(1):
                t = torch.cat([t[:, [0], :, :], t, t[:, [-1], :, :]], dim=1)
            # Generate causal mask if not provided
            if tgt_mask is None:
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
                tgt_mask = tgt_mask.to(tgt.device)

            output = self.decoder(
                t,
                tgt,
                adj,
                None,  # No memory in decoder-only mode
                tgt_mask,
                None,  # No memory mask needed
                tgt_key_padding_mask,
                None   # No memory key padding mask needed
            )

        return self.output_projection(output)

    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate output sequence auto-regressively

        Args:
            src: Input sequence [batch_size, seq_len, node_size, d_model]
            max_len: Maximum length of the generated sequence
            temperature: Sampling temperature (1.0 means greedy)

        Returns:
            Generated sequence [batch_size, max_len, node_size, d_model]
        """
        device = src.device
        batch_size = src.size(0)

        if self.mode == 'encoder_decoder':
            # Encode the source sequence
            memory = self.encoder(src)

            # Initialize decoder input with START token
            decoder_input = self.special_tokens.start_token.unsqueeze(0).unsqueeze(0)
            decoder_input = decoder_input.expand(batch_size, 1, -1, -1)
            decoder_input = decoder_input.to(device)

            # Generate sequence
            for i in range(max_len - 1):
                # Create causal mask
                tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1))
                tgt_mask = tgt_mask.to(device)

                # Get next prediction
                output = self.decoder(
                    decoder_input,
                    memory,
                    tgt_mask=tgt_mask
                )
                next_token = self.output_projection(output[:, -1:, :, :])

                if temperature != 1.0:
                    next_token = next_token / temperature

                # Add predicted token to decoder input
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

                # Check for END token
                if torch.allclose(next_token, self.special_tokens.end_token, rtol=0.1):
                    break

            return decoder_input

        else:  # decoder_only mode
            # Initialize with START token and source sequence
            decoder_input = self.special_tokens.start_token.unsqueeze(0).unsqueeze(0)
            decoder_input = decoder_input.expand(batch_size, 1, -1, -1)
            decoder_input = decoder_input.to(device)
            decoder_input = torch.cat([decoder_input, src], dim=1)

            # Generate sequence
            for i in range(max_len - 1):
                # Create causal mask
                tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1))
                tgt_mask = tgt_mask.to(device)

                # Get next prediction
                output = self.decoder(
                    decoder_input,
                    None,  # No memory in decoder-only mode
                    tgt_mask=tgt_mask
                )
                next_token = self.output_projection(output[:, -1:, :, :])

                if temperature != 1.0:
                    next_token = next_token / temperature

                # Add predicted token to decoder input
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

                # Check for END token
                if torch.allclose(next_token, self.special_tokens.end_token, rtol=0.1):
                    break

            # Remove the initial source sequence from output
            start_idx = src.size(1) + 1  # +1 for START token
            return decoder_input[:, start_idx:]

    def init_weights(self) -> None:
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
