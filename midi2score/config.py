from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from midi2score.model_decoder import DecoderLanguageModelConfig

from midi2score.data_seq2seq import Seq2SeqDataConfig
from midi2score.model_seq2seq import Seq2SeqConfig, EncoderConfig
from midi2score.train_seq2seq import Seq2SeqTrainingConfig


@dataclass(slots=True)
class Seq2SeqProjectConfig:
    model: Seq2SeqConfig
    data: Seq2SeqDataConfig
    training: Seq2SeqTrainingConfig

def load_model_configs(model_section: dict[str, Any]) -> Seq2SeqConfig:
    # load encoder config
    encoder_config_path = Path(model_section["encoder_config_path"])
    with encoder_config_path.open("r", encoding="utf-8") as handle:
        encoder_raw_config = yaml.safe_load(handle)
    if not isinstance(encoder_raw_config, dict):
        raise ValueError("Encoder config contains errors")
    encoder_config = EncoderConfig(**encoder_raw_config)

    # load decoder config
    decoder_config_path = Path(model_section["decoder_config_path"])
    with decoder_config_path.open("r", encoding="utf-8") as handle:
        decoder_raw_config = yaml.safe_load(handle)
    if not isinstance(decoder_raw_config, dict):
        raise ValueError("Decoder config contains errors")
    decoder_model_section = _get_section(decoder_raw_config, "model")
    decoder_config = DecoderLanguageModelConfig(**decoder_model_section)

    return Seq2SeqConfig(encoder_config=encoder_config, decoder_config=decoder_config)


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
        model=load_model_configs(model_section),
        data=Seq2SeqDataConfig(**data_section),
        training=Seq2SeqTrainingConfig(**training_section),
    )


def _get_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = raw_config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")

    return section