from __future__ import annotations

from dataclasses import dataclass, asdict, field
from functools import partial
from typing import TypedDict
import warnings

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, get_worker_info
from pathlib import Path
import json

@dataclass(slots=True)
class Seq2SeqDataConfig:
    # =========================
    # Dataset
    # =========================
    dataset_path: str
    split: str = "training"

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

        if self.split not in {"training", "validation", "test"}:
            raise ValueError("split must be one of training/validation/test")

        if self.bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive")
        
        for i in range(len(self.curriculum_stage_prob)):
            if sum(self.curriculum_stage_prob[i]) != 1.0:
                warnings.warn(f"curriculum_stage_prob[{i}] do not sum to 1.0, normalizing them to sum to 1.0")
                total = sum(self.curriculum_stage_prob[i])
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
    def __init__(self, config: Seq2SeqDataConfig):
        self.config = config

        self.current_stage = 0

        dataset_dict = load_from_disk(config.dataset_path)
        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError("dataset_path must point to DatasetDict")

        self.dataset: HFDataset = dataset_dict[config.split]
        self.lengths = list(self.dataset["lmx_length"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Seq2SeqExample:
        item = self.dataset[index]
        
        noise_keys = ["midi_clean_ids", "midi_light_ids", "midi_heavy_ids"]
        probs = torch.tensor(self.config.curriculum_stage_prob[self.current_stage])
        idx = torch.multinomial(probs, num_samples=1).item()
        selected_key = noise_keys[idx]

        lmx = item["lmx_ids"]
        midi = item[selected_key]

        midi_trimmed = midi[:self.config.max_source_length]
        lmx_trimmed = lmx[:self.config.max_target_length]

        return {
            "encoder_tokens": torch.tensor(midi_trimmed, dtype=torch.long),
            "decoder_tokens": torch.tensor(lmx_trimmed, dtype=torch.long),
        }
    
    def set_stage(self, stage: int):
        if 0 <= stage and stage < len(self.config.curriculum_stage_prob):
            self.current_stage = stage
            self.noise_probs = self.config.curriculum_stage_prob[stage]
            # print(f"Dataset switched to stage: {stage} (Probs: {self.noise_probs})")
        else:
            raise ValueError(f"Stage {stage} is out of bounds for curriculum_stage_prob")

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

    # ✅ 更安全 mask（针对 [T,7]）
    encoder_mask = encoder_tokens[..., 0].eq(pad_token_id)

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
    shuffle: bool | None = None,
):
    dataset = HuggingFaceSeq2SeqDataset(config)

    if shuffle is None:
        shuffle = config.split == "training"

    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": config.num_workers,
        "collate_fn": partial(
            collate_seq2seq_batch,
            pad_token_id=config.pad_token_id,
        ),
    }

    if config.split == "training" and config.length_bucketing:
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
        if config.split == "training":
            generator = torch.Generator().manual_seed(config.shuffle_seed)

        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["generator"] = generator

    return DataLoader(**dataloader_kwargs)