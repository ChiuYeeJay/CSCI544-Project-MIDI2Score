from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
from typing import TypedDict

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
    # Cropping
    # =========================
    random_crop: bool = True
    crop_seed: int = 0

    # =========================
    # Sliding window
    # =========================
    sliding_window_stride: int | None = None

    # =========================
    # Length bucketing
    # =========================
    length_bucketing: bool = False
    bucket_size_multiplier: int = 50

    # =========================
    # Dataloader
    # =========================
    num_workers: int = 0

    # =========================
    # Tokens
    # =========================
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

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

        if self.sliding_window_stride is not None and self.sliding_window_stride <= 0:
            raise ValueError("sliding_window_stride must be positive")

        if self.bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive")

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

        dataset_dict = load_from_disk(config.dataset_path)
        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError("dataset_path must point to DatasetDict")

        self.dataset: HFDataset = dataset_dict[config.split]

        self._crop_generators: dict[int, torch.Generator] = {}
        self._window_index: list[tuple[int, int]] | None = None

        if config.sliding_window_stride is not None:
            self._window_index = self._build_window_index()

    def __len__(self):
        return len(self._window_index) if self._window_index else len(self.dataset)

    def __getitem__(self, index: int) -> Seq2SeqExample:
        raw_index, window_start = self._resolve_index(index)

        item = self.dataset[raw_index]
        midi = item["midi_clean_ids"]
        lmx = item["lmx_ids"]

        # =========================
        # 🎯 计算 start（统一逻辑）
        # =========================
        if window_start is not None:
            # sliding window 情况
            base_start = window_start
        else:
            # random crop（training 时）
            if self.config.random_crop and self.config.split == "training":
                max_start = min(
                        max(len(midi) - self.config.max_source_length, 0),
                        max(len(lmx) - self.config.max_target_length, 0)
                    )

                base_start = int(
                    torch.randint(
                        0,
                        max_start + 1,
                        (1,),
                        generator=self._get_crop_generator()
                    )
                )
            else:
                base_start = None

        # =========================
        # 🎯 ratio 弱对齐
        # =========================
        if base_start is not None:
            ratio = (len(lmx) - 1) / max(len(midi) - 1, 1)

            midi_start = base_start
            lmx_start = int(base_start * ratio)
        
            lmx_start = min(
                lmx_start,
                max(len(lmx) - self.config.max_target_length, 0)
            )
        else:
            midi_start = None
            lmx_start = None

        # =========================
        # 🎯 分别裁剪（不等长）
        # =========================
        midi = self._trim(
            midi,
            midi_start,
            self.config.max_source_length
        )

        lmx = self._trim(
            lmx,
            lmx_start,
            self.config.max_target_length - 1  # 给 EOS 留位置
        )

        # =========================
        # 🎯 加 EOS（decoder）
        # =========================
        lmx = lmx + [self.config.eos_token_id]

        return {
            "encoder_tokens": torch.tensor(midi, dtype=torch.long),
            "decoder_tokens": torch.tensor(lmx, dtype=torch.long),
        }

    def _resolve_index(self, index):
        if self._window_index is None:
            return index, None
        return self._window_index[index]

    def _trim(self, seq, start, max_len):
        if len(seq) <= max_len:
            return seq

        if start is not None:
            return seq[start:start + max_len]

        return seq[:max_len]

    def _build_window_index(self):
        stride = self.config.sliding_window_stride
        windows = []

        for i in range(len(self.dataset)):
            midi = self.dataset[i]["midi_clean_ids"]
            lmx = self.dataset[i]["lmx_ids"]

            src_len = len(midi)
            tgt_len = len(lmx)

            max_src_start = max(src_len - self.config.max_source_length, 0)
            max_tgt_start = max(tgt_len - self.config.max_target_length, 0)

            max_start = max(max_src_start, max_tgt_start)

            if max_start == 0:
                windows.append((i, 0))
                continue

            starts = list(range(0, max_start + 1, stride))
            if starts[-1] != max_start:
                starts.append(max_start)

            windows.extend((i, s) for s in starts)

        return windows

    def _get_crop_generator(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        generator = self._crop_generators.get(worker_id)
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(self.config.crop_seed + 1_000_003 * worker_id)
            self._crop_generators[worker_id] = generator

        return generator


# =========================
# Length Bucketing（完整补上）
# =========================
class LengthBucketBatchSampler(BatchSampler):
    def __init__(self, *, dataset, batch_size, drop_last, seed, bucket_size_multiplier):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size_multiplier = bucket_size_multiplier
        self._epoch = 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1

        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        bucket_size = self.batch_size * self.bucket_size_multiplier

        batches = []
        for i in range(0, len(indices), bucket_size):
            pool = indices[i:i + bucket_size]
            for j in range(0, len(pool), self.batch_size):
                batch = pool[j:j + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        for idx in torch.randperm(len(batches), generator=g):
            yield batches[idx]

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# =========================
# Collate
# =========================
def collate_seq2seq_batch(
    examples: list[Seq2SeqExample],
    *,
    pad_token_id: int,
    bos_token_id: int,
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
    decoder_input = decoder_tokens.clone()
    decoder_input[:, 1:] = decoder_tokens[:, :-1]
    decoder_input[:, 0] = bos_token_id

    labels = decoder_tokens.clone()
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
            bos_token_id=config.bos_token_id,
        ),
    }

    if config.split == "training" and config.length_bucketing:
        dataloader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
            seed=config.crop_seed,
            bucket_size_multiplier=config.bucket_size_multiplier,
        )
    else:
        generator = None
        if config.split == "training":
            generator = torch.Generator().manual_seed(config.crop_seed)

        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["generator"] = generator

    return DataLoader(**dataloader_kwargs)