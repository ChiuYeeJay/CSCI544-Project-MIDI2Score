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
        if self.position_encoding_type not in {"sinusoidal", "rope"}:
            raise ValueError("position_encoding_type must be one of {'sinusoidal', 'rope'}")
    
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
        self.use_rope = config.position_encoding_type == "rope"

        self.embedding = CPWordEmbedding(
            vocab_size_list=config.src_vocab_size_list,
            embbed_size_list=config.src_embedding_size_list,
            d_model=config.d_model,
            pad_token_id=config.pad_token_id,
        )

        self.position_encoding = None
        if not self.use_rope:
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
            rotary_pos_emb=self.use_rope,
        )

    def forward(
        self,
        input_tokens: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        
        embeddings = self.embedding(input_tokens)
        
        if self.position_encoding is not None:
            x = embeddings + self.position_encoding(input_tokens)
        else:
            x = embeddings
        x = self.dropout(x)

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

        # If all labels are ignored (e.g., ultra-short targets), avoid NaN from mean reduction.
        if labels.ne(-100).sum() == 0:
            zero_loss = logits.sum() * 0.0
            return zero_loss, logits

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss, logits
    
    def generate(
        self,
        encoder_input_tokens: Tensor,
        encoder_padding_mask: Tensor | None = None,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """
        Autoregressive generation for seq2seq model.

        Args:
            encoder_input_tokens: [B, T_src, ...]
            encoder_padding_mask: [B, T_src]
            max_length: max generation length
            temperature: sampling temperature
            top_k: optional top-k sampling

        Returns:
            generated_tokens: [B, T_gen]
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                device = encoder_input_tokens.device
                config = self.config.decoder_config

                if max_length is None:
                    max_length = config.max_length

                if encoder_padding_mask is not None:
                    if encoder_padding_mask.dim() == 3:
                        encoder_padding_mask = encoder_padding_mask[..., 0]
                    encoder_padding_mask = encoder_padding_mask.to(torch.bool)

                batch_size = encoder_input_tokens.size(0)

                bos_token_id = config.bos_token_id
                eos_token_id = config.eos_token_id
                pad_token_id = config.pad_token_id

                memory = self.model.encoder(
                    encoder_input_tokens,
                    padding_mask=encoder_padding_mask,
                )

                generated = torch.full(
                    (batch_size, 1),
                    bos_token_id,
                    dtype=torch.long,
                    device=device,
                )

                finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
                past_key_values = None

                for _ in range(max_length):
                    decoder_tokens = generated if past_key_values is None else generated[:, -1:]
                    logits, past_key_values = self.model.decoder(
                        decoder_tokens,
                        memory=memory,
                        memory_padding_mask=encoder_padding_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    next_token_logits = logits[:, -1, :]

                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    if top_k is not None:
                        current_top_k = min(top_k, next_token_logits.size(-1))
                        values, indices = torch.topk(next_token_logits, current_top_k)
                        probs = torch.zeros_like(next_token_logits).scatter_(1, indices, torch.softmax(values, dim=-1))
                    else:
                        probs = torch.softmax(next_token_logits, dim=-1)

                    next_tokens = torch.multinomial(probs, num_samples=1)
                    next_tokens = next_tokens.masked_fill(finished.unsqueeze(-1), pad_token_id)

                    generated = torch.cat([generated, next_tokens], dim=1)
                    finished = finished | (next_tokens.squeeze(-1) == eos_token_id)

                    if finished.all():
                        break

                return generated
        finally:
            if was_training:
                self.train()