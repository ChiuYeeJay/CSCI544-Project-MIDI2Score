from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from x_transformers import Encoder

from midi2score.model_decoder import TransformerDecoderLM, SinusoidalPositionalEncoding, DecoderLanguageModelConfig

@dataclass(slots=True)
class EncoderConfig:
    # CPWord encoder embedding
    src_vocab_size_list: list[int]
    src_embedding_size_list: list[int]
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.05
    activation: str = "swiglu"
    max_length: int = 2048
    position_encoding_type: str = "sinusoidal"

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if len(self.src_vocab_size_list) != len(self.src_embedding_size_list):
            raise ValueError("src_vocab_size_list and src_embedding_size_list must match")
    
    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class Seq2SeqConfig:
    encoder_config: EncoderConfig
    decoder_config: DecoderLanguageModelConfig

    def to_dict(self):
        return asdict(self)


class CPWordEmbedding(nn.Module):
    def __init__(self, vocab_size_list, embbed_size_list, d_model=512, pad_token_id=0):
        super().__init__()
        self.family_embedding = nn.Embedding(vocab_size_list[0], embbed_size_list[0], padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(vocab_size_list[1], embbed_size_list[1], padding_idx=pad_token_id)
        self.pitch_embedding = nn.Embedding(vocab_size_list[2], embbed_size_list[2], padding_idx=pad_token_id)
        self.duration_embedding = nn.Embedding(vocab_size_list[3], embbed_size_list[3], padding_idx=pad_token_id)
        self.program_embedding = nn.Embedding(vocab_size_list[4], embbed_size_list[4], padding_idx=pad_token_id)
        self.tempo_embedding = nn.Embedding(vocab_size_list[5], embbed_size_list[5], padding_idx=pad_token_id)
        self.time_signature_embedding = nn.Embedding(vocab_size_list[6], embbed_size_list[6], padding_idx=pad_token_id)

        self.in_linear = nn.Linear(sum(embbed_size_list), d_model)

    def forward(self, x):
        family = self.family_embedding(x[..., 0])
        position = self.position_embedding(x[..., 1])
        pitch = self.pitch_embedding(x[..., 2])
        duration = self.duration_embedding(x[..., 3])
        program = self.program_embedding(x[..., 4])
        tempo = self.tempo_embedding(x[..., 5])
        ts = self.time_signature_embedding(x[..., 6])

        x = torch.cat([family, position, pitch, duration, program, tempo, ts], dim=-1)
        return self.in_linear(x)


class TransformerEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = CPWordEmbedding(
            vocab_size_list=config.src_vocab_size_list,
            embbed_size_list=config.src_embedding_size_list,
            d_model=config.d_model,
            pad_token_id=config.pad_token_id,
        )

        self.position_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_length)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder = Encoder(
            dim=config.d_model,
            depth=config.num_layers,
            heads=config.nhead,
            ff_mult=config.dim_feedforward // config.d_model,
            attn_dropout=config.dropout,
            ff_dropout=config.dropout,
            use_rmsnorm=True,
            pre_norm=True,
            ff_glu=True,
            ff_swish = True,
            attn_flash=True,
            rotary_pos_emb=False,
        )

    def forward(
        self,
        input_tokens: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        
        embeddings = self.embedding(input_tokens)
        
        x = self.dropout(embeddings + self.position_encoding(input_tokens))

        x_mask = None
        if padding_mask is not None:
            if padding_mask.dim() == 3:
                padding_mask = padding_mask[..., 0]
            x_mask = ~(padding_mask.to(torch.bool))

        hidden_states = self.encoder(x, mask=x_mask)
        return hidden_states


class TransformerSeq2Seq(nn.Module):
    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config.encoder_config)
        self.decoder = TransformerDecoderLM(config.decoder_config)

    def forward(
        self,
        *,
        encoder_input_tokens: Tensor,
        decoder_input_tokens: Tensor,
        encoder_padding_mask: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if encoder_padding_mask is not None:
            if encoder_padding_mask.dim() == 3:
                encoder_padding_mask = encoder_padding_mask[..., 0]
            encoder_padding_mask = encoder_padding_mask.to(torch.bool)

        if decoder_padding_mask is not None:
            decoder_padding_mask = decoder_padding_mask.to(torch.bool)

        memory = self.encoder(
            encoder_input_tokens,
            padding_mask=encoder_padding_mask,
        )

        logits = self.decoder(
            decoder_input_tokens,
            padding_mask=decoder_padding_mask,
            memory=memory,
            memory_padding_mask=encoder_padding_mask,
        )
        return logits


class TransformerForConditionalGeneration(nn.Module):
    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TransformerSeq2Seq(config)

    def forward(
        self,
        *,
        encoder_input_tokens: Tensor,
        decoder_input_tokens: Tensor,
        labels: Tensor | None = None,
        encoder_padding_mask: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
    ):
        if labels is None:
            raise ValueError("labels must be provided for training")

        if decoder_padding_mask is None:
            decoder_padding_mask = decoder_input_tokens.eq(self.config.decoder_config.pad_token_id)

        logits = self.model(
            encoder_input_tokens=encoder_input_tokens,
            decoder_input_tokens=decoder_input_tokens,
            encoder_padding_mask=encoder_padding_mask,
            decoder_padding_mask=decoder_padding_mask,
        )

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss, logits