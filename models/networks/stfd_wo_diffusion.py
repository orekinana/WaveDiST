import math
from typing import Literal

import torch.nn
from ..layers.st_transformer import STTransformer
from ..layers.diffusion import gaussian_diffusion
from ..layers.freq_transform import WaveletNoiseTransformer

import torch.nn.functional as F


class STWODiffusionModel(torch.nn.Module):

    def __init__(
        self,
        dim_feature: int,
        dim_seq: int,
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
        noise_schedule: str = 'linear', 
        diffusion_steps: int = 1000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sigma_small, learn_sigma = False, True
        loss_type = gaussian_diffusion.LossType.RESCALED_MSE
        self.betas = gaussian_diffusion.get_named_beta_schedule(noise_schedule, diffusion_steps)
        self.model_mean_type = gaussian_diffusion.ModelMeanType.START_X
        self.model_var_type=(
            (
                gaussian_diffusion.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gaussian_diffusion.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gaussian_diffusion.ModelVarType.LEARNED_RANGE
        )
        self.diffusion = gaussian_diffusion.GaussianDiffusion(self.betas, self.model_mean_type, self.model_var_type, loss_type)

        self.waveTrans = WaveletNoiseTransformer()

        self.time_embedder = TimestepEmbedder(dim_feature)

        self.fc_l = torch.nn.Linear(dim_seq // 8, dim_seq)
        self.fc_h1 = torch.nn.Linear(dim_seq // 8, dim_seq)
        self.fc_h2 = torch.nn.Linear(dim_seq // 4, dim_seq)
        self.fc_h3 = torch.nn.Linear(dim_seq // 2, dim_seq)

        self.fc_inv = torch.nn.Linear(dim_feature, 1)
        # self.fc = torch.nn.Linear(397, 200)
        # self.fc_inv = torch.nn.Linear(200, 397)


        self.st_transformer = STTransformer(
            dim_feature=dim_feature,
            dim_node=dim_node,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            encoder_attention_type=encoder_attention_type,
            decoder_attention_type=decoder_attention_type,
            mode=mode,
            temporal_pos_embedding=temporal_pos_embedding,
            node_pos_embedding=node_pos_embedding,
            special_tokens_init=special_tokens_init,
        )


    # def forward(self, x, *args, **kwargs,):
    def forward(self, x, adj):

        '''
        x_t_h1: batch_size x node_size x seq_h1
        x_t_h2: batch_size x node_size x seq_h2
        x_t_h3: batch_size x node_size x seq_h3
        x_l: batch_size x node_size x seq_l
        '''
        x_t_h1, x_t_h2, x_t_h3, y_h1, y_h2, y_h3, x_l, t = self.freq_trans(x)
        x_t = torch.cat([
            torch.nn.functional.silu(self.fc_h1(x_t_h1)).transpose(-1, -2).unsqueeze(-1), # batch_size x seq x node_size x 1
            torch.nn.functional.silu(self.fc_h2(x_t_h2)).transpose(-1, -2).unsqueeze(-1), # batch_size x seq x node_size x 1
            torch.nn.functional.silu(self.fc_h3(x_t_h3)).transpose(-1, -2).unsqueeze(-1), # batch_size x seq x node_size x 1
        ], dim=-1) # batch_size x seq x node_size x dim_feature

        y = torch.cat([
            torch.nn.functional.silu(self.fc_h1(y_h1)).transpose(-1, -2).unsqueeze(-1), # batch_size x seq x node_size x 1
            torch.nn.functional.silu(self.fc_h2(y_h2)).transpose(-1, -2).unsqueeze(-1), # batch_size x seq x node_size x 1
            torch.nn.functional.silu(self.fc_h3(y_h3)).transpose(-1, -2).unsqueeze(-1), # batch_size x seq x node_size x 1
        ], dim=-1) # batch_size x seq x node_size x dim_feature

        tgt, src, output = y, x_t, {}
        
        time_embedding = self.time_embedder(t).unsqueeze(1).unsqueeze(1)
        t_l = torch.nn.functional.silu(self.fc_l(x_l)).transpose(-1, -2).unsqueeze(-1).repeat(1, 1, 1, time_embedding.size(-1)) # batch_size x seq x node_size x dim_feature
        time_embedding = time_embedding + t_l

        batch_size, *_ = src.size()
        start_token = self.st_transformer.special_tokens.start_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).to(src.device)
        end_token = self.st_transformer.special_tokens.end_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).to(src.device)
        tgt_input = torch.cat([start_token, tgt, end_token], dim=1)
        tgt_mask = self.st_transformer._generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)

        tgt_hat = self.st_transformer(t=time_embedding, src=src, adj=adj, tgt=tgt_input, tgt_mask=tgt_mask)

        valid_predictions = tgt_hat[:, :-2, :, :] # batch_size x seq x node_size x dim_feature

        valid_predictions = self.fc_inv(valid_predictions)

        return valid_predictions
    
    def freq_trans(self, x):
        # Wavelet Transformer for generate multi layer high freq signals and one low freq signal
        signals = self.waveTrans.transform(x)
        # add noise to multi layer high freq signals
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=x.device)
       
        x_l = signals[0]
        x_t_h1 = signals[1]
        x_t_h2 = signals[2]
        x_t_h3 = signals[3]

        y_h1 = signals[1]
        y_h2 = signals[2]
        y_h3 = signals[3]

        return x_t_h1, x_t_h2, x_t_h3, y_h1, y_h2, y_h3, x_l, t

    def loss_fn(self, y_hat, y, mask):

        return (mask * ((y - y_hat) ** 2)).sum() / mask.sum()
        return self.mean_flat(mask * ((y - y_hat) ** 2)).mean()
    
    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.sum(dim=list(range(1, len(tensor.shape))))


class TimestepEmbedder(torch.nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
