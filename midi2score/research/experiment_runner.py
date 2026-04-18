from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, UTC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from midi2score.config import load_seq2seq_config
from midi2score.data_seq2seq import Seq2SeqDataConfig
from midi2score.model_decoder import DecoderLanguageModelConfig
from midi2score.model_seq2seq import EncoderConfig
from midi2score.train_seq2seq import (
    Seq2SeqTrainingConfig,
    run_seq2seq_training_loop,
)

from midi2score.research.git_utils import (
    collect_git_metadata,
    require_clean_git_worktree,
)

# =========================
# experiment_id 规范
# =========================
_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

_ALLOWED_OVERRIDE_FIELDS = {
    "model": {
        "encoder": {field.name for field in fields(EncoderConfig)},
        "decoder": {field.name for field in fields(DecoderLanguageModelConfig)},
    },
    "data": {field.name for field in fields(Seq2SeqDataConfig)},
    "training": {field.name for field in fields(Seq2SeqTrainingConfig)},
}


# =========================
# 路径结构
# =========================
@dataclass(slots=True)
class ExperimentPaths:
    config_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    log_dir: Path
    tensorboard_log_dir: Path
    summary_path: Path


# =========================
# override解析
# =========================
def parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


# =========================
# 构建实验config
# =========================
def build_experiment_config(
    *,
    base_config_path: str | Path,
    experiment_id: str,
    overrides: dict[str, Any],
    output_root: str | Path = ".",
):
    if not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id may only contain letters, numbers, dot, underscore, and hyphen."
        )

    output_root = Path(output_root)
    paths = _build_experiment_paths(output_root, experiment_id)

    raw_config = _load_raw_config(base_config_path)
    resolved_config = deepcopy(raw_config)

    _apply_overrides(resolved_config, overrides)
    _inject_standardized_output_paths(
        resolved_config,
        paths=paths,
        clear_resume="training.resume_checkpoint_path" not in overrides,
    )

    paths.config_path.parent.mkdir(parents=True, exist_ok=True)

    with paths.config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)

    return paths.config_path, paths, resolved_config


# =========================
# 主入口：运行实验
# =========================
def run_research_experiment(
    *,
    base_config_path,
    experiment_id,
    overrides,
    output_root=".",
    repo_root=".",
    note=None,
    reference_best_validation_loss=None,
    require_clean_git=True,
):
    config_path, paths, resolved_config = build_experiment_config(
        base_config_path=base_config_path,
        experiment_id=experiment_id,
        overrides=overrides,
        output_root=output_root,
    )

    git_metadata = (
        require_clean_git_worktree(repo_root)
        if require_clean_git
        else collect_git_metadata(repo_root)
    )

    started_at = datetime.now(UTC)

    project_config = load_seq2seq_config(config_path)

    result = run_seq2seq_training_loop(
        project_config.model,
        project_config.data,
        project_config.training,
    )

    finished_at = datetime.now(UTC)

    summary = {
        "experiment_id": experiment_id,
        "base_config_path": str(Path(base_config_path).resolve()),
        "resolved_config_path": str(config_path.resolve()),
        "note": note,
        "overrides": overrides,

        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),

        "best_validation_loss": result.best_validation_loss,
        "final_step": result.final_step,
        "elapsed_seconds": result.elapsed_seconds,

        "stopped_due_to_time_budget": getattr(result, "stopped_due_to_time_budget", None),
        "stopped_due_to_early_stopping": getattr(result, "stopped_due_to_early_stopping", None),
        "resumed_from_checkpoint": getattr(result, "resumed_from_checkpoint", None),
        "optimizer_state_loaded": getattr(result, "optimizer_state_loaded", None),

        "device": result.device,

        "checkpoint_path": str(paths.checkpoint_path.resolve()),
        "best_checkpoint_path": str(paths.best_checkpoint_path.resolve()),

        "log_dir": str(paths.log_dir.resolve()),
        "csv_log_dir": str((paths.log_dir / "csv").resolve()),
        "tensorboard_log_dir": str(paths.tensorboard_log_dir.resolve()),

        "summary_path": str(paths.summary_path.resolve()),
        "resolved_config": resolved_config,

        "git": git_metadata,
    }

    if (
        reference_best_validation_loss is not None
        and result.best_validation_loss is not None
    ):
        summary["reference_best_validation_loss"] = reference_best_validation_loss
        summary["delta_to_reference"] = (
            result.best_validation_loss - reference_best_validation_loss
        )

    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)

    with paths.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    return summary


# =========================
# helpers
# =========================
def _build_experiment_paths(output_root: Path, experiment_id: str):
    log_dir = output_root / "logs/research" / experiment_id
    return ExperimentPaths(
        config_path=output_root / "configs/research" / f"{experiment_id}.yaml",
        checkpoint_path=output_root / "artifacts/research" / experiment_id / "latest.pt",
        best_checkpoint_path=output_root / "artifacts/research" / experiment_id / "best.pt",
        log_dir=log_dir,
        tensorboard_log_dir=log_dir / "tensorboard",
        summary_path=output_root / "artifacts/research" / experiment_id / "summary.json",
    )


def _load_raw_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")

    return raw


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]):
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")

        if len(parts) < 2:
            raise ValueError(
                f"Override key {dotted_key!r} must target nested field like model.d_model"
            )

        root = parts[0]

        if root == "model":
            if len(parts) < 3:
                raise ValueError(
                    f"Override key {dotted_key!r} must target nested field like model.encoder.d_model"
                )

            model_config = config.get("model")
            if not isinstance(model_config, dict):
                raise ValueError("Config section 'model' must be a mapping.")

            subsection = parts[1]
            allowed_model_fields = _ALLOWED_OVERRIDE_FIELDS["model"]
            if subsection not in allowed_model_fields:
                raise ValueError(f"Unknown config field: {dotted_key}")

            section_config = model_config.get(subsection)
            if not isinstance(section_config, dict):
                raise ValueError(f"Invalid override path: {dotted_key}")

            cursor = section_config
            for part in parts[2:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    raise ValueError(f"Invalid override path: {dotted_key}")
                cursor = cursor[part]

            leaf = parts[-1]
            if leaf not in allowed_model_fields[subsection]:
                raise ValueError(f"Unknown config field: {dotted_key}")

            cursor[leaf] = value
            continue

        if root not in {"data", "training"}:
            raise ValueError(f"Unknown config section: {root}")

        if len(parts) != 2:
            raise ValueError(
                f"Override key {dotted_key!r} must target nested field like {root}.learning_rate"
            )

        section_config = config.get(root)
        if not isinstance(section_config, dict):
            raise ValueError(f"Config section {root!r} must be a mapping.")

        leaf = parts[1]
        if leaf not in _ALLOWED_OVERRIDE_FIELDS[root]:
            raise ValueError(f"Unknown config field: {dotted_key}")

        section_config[leaf] = value


def _inject_standardized_output_paths(
    config: dict[str, Any],
    *,
    paths: ExperimentPaths,
    clear_resume: bool,
):
    training = config.get("training")

    if not isinstance(training, dict):
        raise ValueError("Config section 'training' must be a mapping.")

    # checkpoint
    training["save_checkpoint_path"] = str(paths.checkpoint_path.resolve())
    training["save_best_checkpoint_path"] = str(paths.best_checkpoint_path.resolve())

    # logging
    training["log_dir"] = str(paths.log_dir.resolve())
    training["tensorboard_log_dir"] = str(paths.tensorboard_log_dir.resolve())

    # resume控制（防止污染实验）
    if clear_resume:
        training["resume_checkpoint_path"] = None