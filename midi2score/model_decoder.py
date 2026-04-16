# copy from https://github.com/daniellaah/MIDI2Score/blob/main/midi2score/model.py
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True)
class DecoderLanguageModelConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    positional_encoding: str = "sinusoidal"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.activation not in {"relu", "gelu", "swiglu", "geglu"}:
            raise ValueError("activation must be relu, gelu, swiglu, or geglu.")
        if self.positional_encoding not in {"sinusoidal", "rope"}:
            raise ValueError("positional_encoding must be sinusoidal or rope.")
        if self.positional_encoding == "rope" and (self.d_model // self.nhead) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.encoding[:, : tokens.size(1)]


class ZeroPositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: Tensor) -> Tensor:
        return torch.zeros(
            (1, tokens.size(1), self.d_model),
            dtype=torch.float32,
            device=tokens.device,
        )


def build_positional_encoding(positional_encoding: str, d_model: int, max_length: int) -> nn.Module:
    if positional_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_length)
    if positional_encoding == "rope":
        return ZeroPositionalEncoding(d_model)
    raise ValueError(f"Unsupported positional_encoding {positional_encoding!r}.")


def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


def _activation(name: str):
    return torch.relu if name == "relu" else torch.nn.functional.gelu


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_norm(d_model: int) -> nn.Module:
    return RMSNorm(d_model)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int, *, base: float = 10_000.0) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = positions * inv_freq
        self.register_buffer("cos", angles.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", angles.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, sequence_length: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        cos = self.cos[:, :, :sequence_length].to(device=device, dtype=dtype)
        sin = self.sin[:, :, :sequence_length].to(device=device, dtype=dtype)
        return cos, sin


def apply_rotary_embedding(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, dim_feedforward: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.activation_name = activation
        hidden_dim = dim_feedforward * 2 if activation in {"swiglu", "geglu"} else dim_feedforward
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.linear1(x)
        if self.activation_name == "swiglu":
            value, gate = hidden.chunk(2, dim=-1)
            hidden = value * F.silu(gate)
        elif self.activation_name == "geglu":
            value, gate = hidden.chunk(2, dim=-1)
            hidden = value * F.gelu(gate)
        else:
            hidden = _activation(self.activation_name)(hidden)
        return self.linear2(self.dropout(hidden))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dropout: float,
        positional_encoding: str,
        max_length: int,
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = positional_encoding == "rope"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = dropout
        self.rotary = RotaryEmbedding(self.head_dim, max_length=max_length) if self.use_rope else None

    def forward(self, x: Tensor, *, tgt_padding_mask: Tensor | None = None) -> Tensor:
        batch_size, sequence_length, d_model = x.shape
        query = self.q_proj(x).view(batch_size, sequence_length, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, sequence_length, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, sequence_length, self.nhead, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            cos, sin = self.rotary(
                sequence_length,
                device=x.device,
                dtype=query.dtype,
            )
            query = apply_rotary_embedding(query, cos, sin)
            key = apply_rotary_embedding(key, cos, sin)
        attention_mask = None
        use_causal_flag = tgt_padding_mask is None
        if tgt_padding_mask is not None:
            attention_mask = torch.zeros(
                (batch_size, 1, sequence_length, sequence_length),
                dtype=query.dtype,
                device=x.device,
            )
            attention_mask = attention_mask.masked_fill(
                build_causal_mask(sequence_length, device=x.device).unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
            attention_mask = attention_mask.masked_fill(
                tgt_padding_mask[:, None, None, :],
                float("-inf"),
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=use_causal_flag,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
        return self.out_proj(output)


class CrossAttention(nn.Module):
    def __init__(self, *, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = dropout

    def forward(
        self,
        x: Tensor,
        *,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, target_length, d_model = x.shape
        source_length = memory.size(1)
        query = self.q_proj(x).view(batch_size, target_length, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_proj(memory).view(batch_size, source_length, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_proj(memory).view(batch_size, source_length, self.nhead, self.head_dim).transpose(1, 2)

        attention_mask = None
        if memory_padding_mask is not None:
            attention_mask = torch.zeros(
                (batch_size, 1, target_length, source_length),
                dtype=query.dtype,
                device=x.device,
            )
            attention_mask = attention_mask.masked_fill(
                memory_padding_mask[:, None, None, :],
                float("-inf"),
            )

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, target_length, d_model)
        return self.out_proj(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        positional_encoding: str,
        max_length: int,
    ) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            positional_encoding=positional_encoding,
            max_length=max_length,
        )
        self.cross_attn = CrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.feedforward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = build_norm(d_model)
        self.norm2 = build_norm(d_model)
        self.norm3 = build_norm(d_model)

    def forward(
        self,
        tgt: Tensor,
        *,
        memory: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = tgt
        residual_input = self.norm1(x)
        self_attn_output = self.self_attn(
            residual_input,
            tgt_padding_mask=tgt_padding_mask,
        )
        x = x + self.dropout1(self_attn_output)
        if memory is not None:
            cross_attn_input = self.norm3(x)
            cross_attn_output = self.cross_attn(
                cross_attn_input,
                memory=memory,
                memory_padding_mask=memory_padding_mask,
            )
            x = x + self.dropout2(cross_attn_output)
            ff_input = self.norm2(x)
        else:
            ff_input = self.norm2(x)
        ff_output = self.feedforward(ff_input)
        x = x + self.dropout3(ff_output)
        return x


class TransformerDecoderStack(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        positional_encoding: str,
        max_length: int,
    ) -> None:
        super().__init__()
        layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            max_length=max_length,
        )
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = build_norm(d_model)

    def forward(
        self,
        tgt: Tensor,
        *,
        memory: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory=memory,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=memory_padding_mask,
            )
        return self.norm(x)


class TransformerDecoderLM(nn.Module):
    def __init__(self, config: DecoderLanguageModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_encoding = build_positional_encoding(
            config.positional_encoding,
            config.d_model,
            config.max_length,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.decoder = TransformerDecoderStack(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            positional_encoding=config.positional_encoding,
            max_length=config.max_length,
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.reset_parameters()
        self.output_projection.weight = self.tgt_embedding.weight

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def decode(self, input_tokens: Tensor) -> Tensor:
        embeddings = self.tgt_embedding(input_tokens) * math.sqrt(self.config.d_model)
        return self.dropout(embeddings + self.position_encoding(input_tokens))

    def forward(
        self,
        input_tokens: Tensor,
        *,
        padding_mask: Tensor | None = None,
        memory: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        decoded_inputs = self.decode(input_tokens)
        hidden_states = self.decoder(
            decoded_inputs,
            memory=memory,
            tgt_padding_mask=padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        return self.output_projection(hidden_states)
