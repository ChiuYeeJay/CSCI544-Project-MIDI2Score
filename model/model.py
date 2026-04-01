# CONFIGURATION
from transformers import PretrainedConfig

class MyModelConfig(PretrainedConfig):
    model_type = "midi2score"

    def __init__(
        self,

        # vocab
        src_vocab_size=6400,   # MIDI
        tgt_vocab_size=6400,   # LMX

        # model size
        hidden_size=512,
        intermediate_size=None,

        # layers
        num_encoder_layers=8,
        num_decoder_layers=4,

        # attention
        num_attention_heads=8,
        num_key_value_heads=2,

        # dropout
        dropout=0.1,
        hidden_act="silu",

        # norm / rope
        rms_norm_eps=1e-5,
        rope_theta=1e6,

        # position
        max_source_positions=2048,
        max_target_positions=2048,

        # tokens
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,

        # performance
        flash_attn=True,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.dropout = dropout
        self.hidden_act = hidden_act

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.flash_attn = flash_attn


import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union, Dict
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

#RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(orig_dtype)
    
#RoPE positional encoding
def precompute_freqs_cis(
    dim: int,
    max_position_embeddings: int,
    rope_theta: float = 1e6,
    rope_scaling: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cosine and sine tables.

    Args:
        dim: head_dim, usually hidden_size // num_attention_heads
        max_position_embeddings: max sequence length for this module
                                 (encoder用max_source_positions, decoder用max_target_positions)
        rope_theta: RoPE base
        rope_scaling: optional scaling config, usually None for now

    Returns:
        freqs_cos: [max_position_embeddings, dim]
        freqs_sin: [max_position_embeddings, dim]
    """

    # 1) 先计算基础频率，只保留偶数维的一半频率
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2).float() / dim)
    )
    attn_factor = 1.0

    # 2) 可选：长上下文RoPE缩放。你当前项目可以先不启用
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", max_position_embeddings)
        factor = rope_scaling.get("factor", 1.0)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if max_position_embeddings / orig_max > 1.0:
            def inv_dim(beta: float) -> float:
                return (
                    dim * math.log(orig_max / (beta * 2 * math.pi))
                ) / (2 * math.log(rope_theta))

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            ramp = torch.clamp(
                (torch.arange(dim // 2).float() - low) / max(high - low, 1e-3),
                0,
                1,
            )
            inv_freq = inv_freq * (1 - ramp + ramp / factor)

    # 3) 生成位置索引 [0, 1, 2, ..., max_position_embeddings - 1]
    positions = torch.arange(max_position_embeddings)

    # 4) 位置 × 频率，得到每个位置对应的旋转角
    freqs = torch.outer(positions, inv_freq).float()  # [max_pos, dim//2]

    # 5) 扩展到完整head_dim，因为rotate_half后需要和q/k同维度相乘
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, dim]

    # 6) 预计算 cos/sin 表，后面attention里直接索引使用
    freqs_cos = torch.cos(emb) * attn_factor
    freqs_sin = torch.sin(emb) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [max_seq_len, head_dim]
    position_ids: [batch, seq_len]
    """

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 根据 position_ids 取对应位置
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, dim]
        sin = sin[position_ids]
    else:
        cos = cos[: q.shape[-2]]
        sin = sin[: q.shape[-2]]

    # broadcast 到 [batch, heads, seq_len, dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # 应用 RoPE
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for GQA (Grouped Query Attention).

    Args:
        x: Tensor of shape [bsz, seq_len, num_kv_heads, head_dim]
        n_rep: replication factor = num_attention_heads // num_kv_heads

    Returns:
        Tensor of shape [bsz, seq_len, num_attention_heads, head_dim]
    """

    # 如果不需要扩展，直接返回
    if n_rep == 1:
        return x

    bsz, seq_len, num_kv_heads, head_dim = x.shape

    # [bsz, seq, kv_heads, head_dim]
    # → [bsz, seq, kv_heads, 1, head_dim]
    x = x.unsqueeze(3)

    # → [bsz, seq, kv_heads, n_rep, head_dim]
    x = x.expand(bsz, seq_len, num_kv_heads, n_rep, head_dim)

    # → [bsz, seq, kv_heads * n_rep, head_dim]
    x = x.reshape(bsz, seq_len, num_kv_heads * n_rep, head_dim)

    return x

# ATTENTION
class Attention(nn.Module):
    """
    Unified attention module for:
    1) encoder self-attention
    2) decoder self-attention
    3) decoder cross-attention

    Shapes:
        hidden_states:      [bsz, tgt_len, hidden_size]
        key_value_states:   [bsz, src_len, hidden_size] or None
        attention_mask:     [bsz, src_len] or None
        position_embeddings:(cos, sin), each [max_len, head_dim]
        position_ids:       [bsz, tgt_len] or None
    """

    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0

        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

        # Q always uses all query heads
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )

        # K/V may use fewer kv heads (GQA)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash_attn = (
            hasattr(F, "scaled_dot_product_attention") and config.flash_attn
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
        use_rope: bool = True,
    ):
        """
        Args:
            hidden_states:
                query states. [bsz, tgt_len, hidden_size]

            position_embeddings:
                (cos, sin), each [max_len, head_dim]
                Used only when use_rope=True.

            attention_mask:
                padding mask over keys. [bsz, src_len]
                1 means keep, 0 means mask.

            key_value_states:
                None -> self-attention
                Tensor -> cross-attention, K/V from encoder output

            position_ids:
                [bsz, tgt_len], used for RoPE indexing

            past_key_value:
                only meaningful for decoder self-attention caching

            use_cache:
                whether to return new kv cache

            is_causal:
                True for decoder self-attention
                False for encoder self-attention / cross-attention

            use_rope:
                True for self-attention
                False for cross-attention
        """
        bsz, tgt_len, _ = hidden_states.shape
        is_self_attention = key_value_states is None

        # -------------------------
        # 1) Build Q / K / V
        # -------------------------
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, tgt_len, self.num_attention_heads, self.head_dim
        )

        if is_self_attention:
            # self-attention
            key_value_states = hidden_states

        src_len = key_value_states.shape[1]

        key_states = self.k_proj(key_value_states)
        value_states = self.v_proj(key_value_states)

        key_states = key_states.view(
            bsz, src_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, src_len, self.num_key_value_heads, self.head_dim
        )



        # -------------------------
        # 3) KV cache
        # -------------------------
        # Usually cache is used for decoder self-attention, not cross-attention.
        
        
        if past_key_value is not None and is_self_attention:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
            
        present_key_value = (key_states, value_states) if use_cache else None

        # -------------------------
        # 4) GQA repeat + transpose
        # query: [bsz, heads, tgt_len, head_dim]
        # key:   [bsz, heads, src_len, head_dim]
        # value: [bsz, heads, src_len, head_dim]
        # -------------------------
        query_states = query_states.transpose(1, 2)
        key_states = repeat_kv(key_states, self.n_rep).transpose(1, 2)
        value_states = repeat_kv(value_states, self.n_rep).transpose(1, 2)

        # ✅ 正确位置：transpose之后
        if use_rope and is_self_attention:
            if position_embeddings is None:
                raise ValueError("position_embeddings must be provided when use_rope=True")

            cos, sin = position_embeddings

            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
            )

        kv_seq_len = key_states.shape[-2]

        # -------------------------
        # 5) Flash attention path
        # -------------------------
        can_use_flash = (
            self.flash_attn
            and past_key_value is None
            and query_states.is_cuda == key_states.is_cuda == value_states.is_cuda
        )

        if can_use_flash:
            # Build additive mask if needed
            attn_mask_for_flash = None
            if attention_mask is not None:
                # [bsz, src_len] -> [bsz, 1, tgt_len, src_len]
                attn_mask_for_flash = attention_mask[:, None, None, :].to(query_states.dtype)
                attn_mask_for_flash = (1.0 - attn_mask_for_flash) * torch.finfo(query_states.dtype).min

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask_for_flash,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # -------------------------
            # 6) Standard attention path
            # -------------------------
            attn_scores = torch.matmul(
                query_states, key_states.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)

            # causal mask for decoder self-attention only
            if is_causal:
                causal_mask = torch.triu(
                    torch.full(
                        (tgt_len, kv_seq_len),
                        float("-inf"),
                        device=attn_scores.device,
                        dtype=attn_scores.dtype,
                    ),
                    diagonal=1 + (kv_seq_len - tgt_len),
                )
                attn_scores = attn_scores + causal_mask

            # padding mask over source/key side
            if attention_mask is not None:
                # [bsz, src_len] -> [bsz, 1, 1, src_len]
                expanded_mask = attention_mask[:, None, None, :].to(attn_scores.dtype)
                expanded_mask = (1.0 - expanded_mask) * torch.finfo(attn_scores.dtype).min
                attn_scores = attn_scores + expanded_mask

            attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(query_states)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)

        # -------------------------
        # 7) Merge heads
        # -------------------------
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.hidden_size)
        attn_output = self.resid_dropout(self.o_proj(attn_output))

        return attn_output, present_key_value
    

class FeedForward(nn.Module):
    """
    SwiGLU / gated FFN for midi2score encoder-decoder model.

    Input:
        x: [batch_size, seq_len, hidden_size]

    Output:
        [batch_size, seq_len, hidden_size]
    """

    def __init__(self, config):
        super().__init__()

        # 1. 读取 hidden size
        hidden_size = config.hidden_size

        # 2. intermediate size
        # 如果 config 没有显式给，就按 LLaMA / MiniMind 风格自动计算
        if getattr(config, "intermediate_size", None) is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 63) // 64)
        else:
            intermediate_size = config.intermediate_size

        # 3. 检查激活函数是否合法
        if config.hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)

        # 4. SwiGLU / gated FFN
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))   # [B, T, intermediate]
        up = self.up_proj(x)                    # [B, T, intermediate]
        x = gate * up                           # gated interaction
        x = self.down_proj(x)                   # [B, T, hidden]
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x,
        position_embeddings,
        attention_mask=None,
        position_ids=None
    ):
        # Self-Attention
        residual = x
        x, _ = self.self_attn(
            hidden_states=self.norm1(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_causal=False,
            use_rope=True
        )
        x = x + residual

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
    
# DECODER BLOCK
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.cross_attn = Attention(config)

        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm3 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = FeedForward(config)

    def forward(
        self,
        x,
        encoder_hidden_states=None,          # (允许None）
        position_embeddings=None,
        attention_mask=None,
        encoder_attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    ):
        # =========================
        # 1. Self-Attention（始终执行）
        # =========================
        residual = x

        x, present_kv = self.self_attn(
            hidden_states=self.norm1(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            is_causal=True,
            use_rope=True,
        )

        x = x + residual

        # =========================
        # 2. Cross-Attention（条件执行）
        # =========================
        if encoder_hidden_states is not None:
            residual = x

            x, _ = self.cross_attn(
                hidden_states=self.norm2(x),
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_embeddings=None,
                position_ids=None,
                is_causal=False,
                use_rope=False,
            )

            x = x + residual

        # =========================
        # 3. FFN
        # =========================
        x = x + self.mlp(self.norm3(x))

        return x, present_kv
    
# ENCODER 
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # embedding（MIDI）
        self.embed_tokens = nn.Embedding(
            config.src_vocab_size,
            config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

        # encoder layers
        self.layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.num_encoder_layers)
        ])

        # final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE（encoder用source长度）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_source_positions,
            rope_theta=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids: [bsz, src_len]
        """

        bsz, seq_len = input_ids.shape

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float32)

        # embedding
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # RoPE position embeddings
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.cumsum(dim=1) - 1
                position_ids = position_ids.clamp(min=0)
            else:
                position_ids = torch.arange(seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            
        position_embeddings = (
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len],
        )

        # stack encoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states   # [bsz, src_len, hidden]
    
# DECODER 
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # embedding（LMX）
        self.embed_tokens = nn.Embedding(
            config.tgt_vocab_size,
            config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

        # decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.num_decoder_layers)
        ])

        # final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE（decoder用target长度）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_target_positions,
            rope_theta=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        encoder_hidden_states=None,     # (支持None）
        attention_mask=None,
        encoder_attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        input_ids: [bsz, tgt_len]
        """

        bsz, seq_len = input_ids.shape
        position_embeddings = (self.freqs_cos, self.freqs_sin)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float32)

        # embedding
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # RoPE
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        # KV cache准备
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        presents = []

        # stack decoder layers
        for layer, past_kv in zip(self.layers, past_key_values):

            hidden_states, present = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,   # 🔥 控制 cross-attn
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            presents.append(present)

        # final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents
    

class Midi2ScoreModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # 🎹 Encoder（MIDI）
        self.encoder = Encoder(config)

        # 🎼 Decoder（LMX）
        self.decoder = Decoder(config)

    def forward(
        self,
        encoder_input_ids=None,
        decoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        # =========================
        # 1️⃣ Encoder
        # =========================
        encoder_hidden_states = None

        if encoder_input_ids is not None:
            encoder_hidden_states = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
            )

        # =========================
        # 2️⃣ Decoder
        # =========================
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(dtype=torch.float32)

        decoder_outputs, presents = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,  # 🔥 关键
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return {
            "last_hidden_state": decoder_outputs,
            "past_key_values": presents,
            "encoder_hidden_states": encoder_hidden_states,
        }
    
    
class Midi2ScoreForConditionalGeneration(PreTrainedModel):
    config_class = MyModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = Midi2ScoreModel(config)

        # 🎯 LM Head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.tgt_vocab_size,
            bias=False
        )

        # 🔥 weight tying
        self.model.decoder.embed_tokens.weight = self.lm_head.weight

    # =========================
    # 🔧 shift_right（给 seq2seq 用）
    # =========================
    def _shift_right(self, input_ids):
        start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        shifted = input_ids.clone()
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[:, 0] = start_token_id

        shifted = shifted.masked_fill(shifted == -100, pad_token_id)
        return shifted

    # =========================
    # 🔥 forward
    # =========================
    def forward(
        self,
        encoder_input_ids=None,
        decoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        # =========================
        #  判断训练模式
        # =========================
        is_seq2seq = encoder_input_ids is not None

        # =========================
        #  自动构造 decoder_input_ids
        # =========================
        if decoder_input_ids is None and labels is not None:
            if is_seq2seq:
                # Stage B：shift_right
                decoder_input_ids = self._shift_right(labels)
            else:
                # Stage A：GPT-style
                decoder_input_ids = labels

        # =========================
        #  自动构造 decoder_attention_mask
        # =========================
        if decoder_attention_mask is None and decoder_input_ids is not None:
            decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).long()

        # =========================
        #  修正 encoder mask（decoder-only 不需要）
        # =========================
        encoder_attention_mask = (
            encoder_attention_mask if is_seq2seq else None
        )

        # =========================
        # 3️⃣ Backbone
        # =========================
        outputs = self.model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs["last_hidden_state"]

        # =========================
        # 4️⃣ LM Head
        # =========================
        logits = self.lm_head(hidden_states)

        # =========================
        # 5️⃣ Loss（两阶段分开！）
        # =========================
        loss = None
        if labels is not None:
            if is_seq2seq:
                # ✅ Stage B（不shift）
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
            else:
                # ✅ Stage A（shift）
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100
                )

        # =========================
        # 6️⃣ 输出
        # =========================
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "encoder_hidden_states": outputs["encoder_hidden_states"],
        }