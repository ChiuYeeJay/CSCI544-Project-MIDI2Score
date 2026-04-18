from __future__ import annotations

from dataclasses import dataclass, asdict, field
from functools import partial
from typing import Literal, TypedDict
import warnings

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset
from pathlib import Path
import json


NOISE_FIELDS = {
    "clean": "midi_clean_ids",
    "light": "midi_light_ids",
    "heavy": "midi_heavy_ids",
}

@dataclass(slots=True)
class Seq2SeqDataConfig:
    # =========================
    # Dataset
    # =========================
    dataset_path: str

    # =========================
    # Sequence length
    # =========================
    max_source_length: int = 512
    max_target_length: int = 512

    # =========================
    # Tokenizer
    # =========================
    tokenizer_path: str | None = None

    # =========================
    # Length bucketing
    # =========================
    length_bucketing: bool = False
    bucket_size_multiplier: int = 50
    bucketing_mode: Literal["target_only", "source_only", "mixed"] = "mixed"
    source_length_weight: float = 0.2

    # =========================
    # Dataloader
    # =========================
    num_workers: int = 0
    shuffle_seed: int = 0

    # =========================
    # Tokens
    # =========================
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # =========================
    # Curriculum Learning
    # =========================
    # [no_noise, light_noise, heavy_noise]
    curriculum_stage_prob: list[list[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.25, 0.75, 0.0],
            [0.1, 0.4, 0.5]
        ]
    )

    # =========================
    # Validation
    # =========================
    def __post_init__(self) -> None:
        if self.max_source_length < 2:
            raise ValueError("max_source_length must be >= 2")

        if self.max_target_length < 2:
            raise ValueError("max_target_length must be >= 2")

        if self.bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive")
        if self.bucketing_mode not in {"target_only", "source_only", "mixed"}:
            raise ValueError("bucketing_mode must be one of target_only/source_only/mixed")
        if self.source_length_weight < 0:
            raise ValueError("source_length_weight must be non-negative")
        
        for i in range(len(self.curriculum_stage_prob)):
            total = sum(self.curriculum_stage_prob[i])
            if total <= 0.0:
                warnings.warn(
                    f"curriculum_stage_prob[{i}] sums to {total}, replacing with uniform distribution"
                )
                n = len(self.curriculum_stage_prob[i])
                self.curriculum_stage_prob[i] = [1.0 / n] * n
            elif abs(total - 1.0) > 1e-8:
                warnings.warn(f"curriculum_stage_prob[{i}] do not sum to 1.0, normalizing them to sum to 1.0")
                self.curriculum_stage_prob[i] = [p / total for p in self.curriculum_stage_prob[i]]

    # =========================
    # 🔥 tokenizer vocab size
    # =========================
    def tokenizer_vocab_size(self) -> int | None:
        if self.tokenizer_path is None:
            return None
        tokenizer = json.loads(Path(self.tokenizer_path).read_text(encoding="utf-8"))
        return len(tokenizer["model"]["vocab"])

    # =========================
    # 🔥 serialization（修复你报错）
    # =========================
    def to_dict(self):
        return asdict(self)

# =========================
# Example / Batch
# =========================
class Seq2SeqExample(TypedDict):
    encoder_tokens: Tensor
    decoder_tokens: Tensor


@dataclass(slots=True)
class Seq2SeqBatch:
    encoder_input_tokens: Tensor
    encoder_padding_mask: Tensor
    decoder_input_tokens: Tensor
    labels: Tensor

    def to(self, device):
        return Seq2SeqBatch(
            encoder_input_tokens=self.encoder_input_tokens.to(device),
            encoder_padding_mask=self.encoder_padding_mask.to(device),
            decoder_input_tokens=self.decoder_input_tokens.to(device),
            labels=self.labels.to(device),
        )


# =========================
# Dataset
# =========================
class HuggingFaceSeq2SeqDataset(Dataset[Seq2SeqExample]):
    def __init__(
        self,
        config: Seq2SeqDataConfig,
        split: Literal["training", "validation", "test"] = "training",
    ):
        self.config = config
        self.split = split

        self.current_stage = 0

        if self.split not in {"training", "validation", "test"}:
            raise ValueError("split must be one of training/validation/test")

        dataset_dict = load_from_disk(config.dataset_path)
        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError("dataset_path must point to DatasetDict")

        if self.split not in dataset_dict:
            raise ValueError(
                f"Dataset split {self.split!r} not found. Available splits: {list(dataset_dict.keys())}"
            )

        self.dataset: HFDataset = dataset_dict[self.split]
        self.noise_variants = ["clean", "light", "heavy"]

        required_columns = {
            "lmx_ids",
            "midi_clean_ids",
            "midi_light_ids",
            "midi_heavy_ids",
            "source_length_clean",
            "source_length_light",
            "source_length_heavy",
            "target_length_clean",
            "target_length_light",
            "target_length_heavy",
            "lmx_cutoff_clean",
            "lmx_cutoff_light",
            "lmx_cutoff_heavy",
        }
        missing = sorted(required_columns.difference(self.dataset.column_names))
        if missing:
            raise ValueError(
                "Dataset schema mismatch. Expected truncated dataset columns are missing: "
                f"{missing}"
            )

        self.source_length_tensors: dict[str, Tensor] = {}
        self.target_length_tensors: dict[str, Tensor] = {}

        for variant in self.noise_variants:
            source_length_field = f"source_length_{variant}"
            target_length_field = f"target_length_{variant}"

            source_lengths = torch.tensor(self.dataset[source_length_field], dtype=torch.float32)
            target_lengths = torch.tensor(self.dataset[target_length_field], dtype=torch.float32)

            self.source_length_tensors[variant] = source_lengths.clamp(max=float(config.max_source_length))
            self.target_length_tensors[variant] = target_lengths.clamp(max=float(config.max_target_length))

        self.lengths: list[float] = []
        self.stage_probs: list[float] = []
        self.noise_probs = torch.empty(0)
        self.set_stage(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Seq2SeqExample:
        item = self.dataset[index]

        idx = torch.multinomial(self.noise_probs, num_samples=1).item()
        variant = self.noise_variants[idx]
        selected_key = NOISE_FIELDS[variant]

        midi = item[selected_key]
        lmx = item["lmx_ids"]
        lmx_cutoff = int(item[f"lmx_cutoff_{variant}"])
        lmx = lmx[:lmx_cutoff]

        midi_trimmed = midi[:self.config.max_source_length]
        lmx_trimmed = lmx[:self.config.max_target_length]

        return {
            "encoder_tokens": torch.tensor(midi_trimmed, dtype=torch.long),
            "decoder_tokens": torch.tensor(lmx_trimmed, dtype=torch.long),
        }
    
    def set_stage(self, stage: int):
        if 0 <= stage and stage < len(self.config.curriculum_stage_prob):
            self.current_stage = stage
            probs = list(self.config.curriculum_stage_prob[stage])

            if len(probs) != len(self.noise_variants):
                raise ValueError(
                    "curriculum_stage_prob must match 3 noise variants (clean/light/heavy)."
                )

            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            prob_sum = float(probs_tensor.sum().item())
            if prob_sum <= 0.0:
                probs_tensor = torch.full((len(self.noise_variants),), 1.0 / len(self.noise_variants))
            else:
                probs_tensor = probs_tensor / prob_sum

            self.stage_probs = probs_tensor.tolist()
            self.noise_probs = probs_tensor
            self._refresh_bucket_lengths()
        else:
            raise ValueError(f"Stage {stage} is out of bounds for curriculum_stage_prob")

    def _refresh_bucket_lengths(self) -> None:
        n_items = len(self.dataset)
        expected_source = torch.zeros(n_items, dtype=torch.float32)
        expected_target = torch.zeros(n_items, dtype=torch.float32)

        for prob, variant in zip(self.stage_probs, self.noise_variants):
            expected_source += float(prob) * self.source_length_tensors[variant]
            expected_target += float(prob) * self.target_length_tensors[variant]

        if self.config.bucketing_mode == "target_only":
            bucket_lengths = expected_target
        elif self.config.bucketing_mode == "source_only":
            bucket_lengths = expected_source
        else:
            bucket_lengths = expected_target + (self.config.source_length_weight * expected_source)

        self.lengths = bucket_lengths.tolist()

# =========================
# Length Bucketing
# =========================
class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: HuggingFaceSeq2SeqDataset,
        batch_size: int,
        bucket_size_multiplier: int = 50,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size_multiplier = bucket_size_multiplier
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        
        # shuffle global order of examples
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        bucket_size = self.batch_size * self.bucket_size_multiplier
        batches: list[list[int]] = []

        for start in range(0, len(indices), bucket_size):
            pool = indices[start : start + bucket_size]
            
            # sort the pool by length in descending order
            pool.sort(key=lambda x: self.dataset.lengths[x], reverse=True)
            
            # split the pool into batches
            for i in range(0, len(pool), self.batch_size):
                batch = pool[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            shuffled_batch_indices = torch.randperm(len(batches), generator=generator).tolist()
            for batch_idx in shuffled_batch_indices:
                yield batches[batch_idx]
        else:
            for batch in batches:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# =========================
# Collate
# =========================
def collate_seq2seq_batch(
    examples: list[Seq2SeqExample],
    *,
    pad_token_id: int,
) -> Seq2SeqBatch:

    encoder_tokens = pad_sequence(
        [e["encoder_tokens"] for e in examples],
        batch_first=True,
        padding_value=pad_token_id,
    )

    decoder_tokens = pad_sequence(
        [e["decoder_tokens"] for e in examples],
        batch_first=True,
        padding_value=pad_token_id,
    )

    # shift right
    decoder_input = decoder_tokens[:, :-1].clone()

    labels = decoder_tokens[:, 1:].clone()
    labels[labels == pad_token_id] = -100

    if encoder_tokens.dim() == 3:
        encoder_mask = encoder_tokens[..., 0].eq(pad_token_id)
    else:
        raise ValueError(f"Unexpected encoder token shape: {tuple(encoder_tokens.shape)}")

    return Seq2SeqBatch(
        encoder_input_tokens=encoder_tokens,
        encoder_padding_mask=encoder_mask,
        decoder_input_tokens=decoder_input,
        labels=labels,
    )


# =========================
# DataLoader
# =========================
def build_seq2seq_dataloader(
    config: Seq2SeqDataConfig,
    *,
    batch_size: int,
    split: Literal["training", "validation", "test"] = "training",
    shuffle: bool | None = None,
):
    dataset = HuggingFaceSeq2SeqDataset(config, split=split)

    if shuffle is None:
        shuffle = split == "training"

    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": config.num_workers,
        "collate_fn": partial(
            collate_seq2seq_batch,
            pad_token_id=config.pad_token_id,
        ),
    }

    if split == "training" and config.length_bucketing:
        print("Using LengthBucketBatchSampler for training dataloader.")
        dataloader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            bucket_size_multiplier=config.bucket_size_multiplier,
            seed=config.shuffle_seed,
            shuffle=shuffle,
            drop_last=False,
        )
    else:
        generator = None
        if split == "training":
            generator = torch.Generator().manual_seed(config.shuffle_seed)

        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["generator"] = generator

    return DataLoader(**dataloader_kwargs)