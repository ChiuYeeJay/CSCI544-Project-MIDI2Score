from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from midi2score.data_seq2seq import Seq2SeqDataConfig
from midi2score.model_seq2seq import Seq2SeqConfig
from midi2score.train_seq2seq import Seq2SeqTrainingConfig


# =========================
# 项目级 config（整合三部分）
# =========================
@dataclass(slots=True)
class Seq2SeqProjectConfig:
    model: Seq2SeqConfig
    data: Seq2SeqDataConfig
    training: Seq2SeqTrainingConfig


# =========================
# 主入口：加载 YAML config
# =========================
def load_seq2seq_config(path: str | Path) -> Seq2SeqProjectConfig:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")

    model_section = _get_section(raw_config, "model")
    data_section = _get_section(raw_config, "data")
    training_section = _get_section(raw_config, "training")

    return Seq2SeqProjectConfig(
        model=Seq2SeqConfig(**model_section),
        data=Seq2SeqDataConfig(**data_section),
        training=Seq2SeqTrainingConfig(**training_section),
    )


# =========================
# helper：安全获取子配置
# =========================
def _get_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = raw_config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")

    return section