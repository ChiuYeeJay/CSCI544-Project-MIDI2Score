from __future__ import annotations

import argparse
import json
from pathlib import Path

from midi2score.config import load_seq2seq_config   # ⚠️ 你需要有这个函数
from midi2score.research import parse_override_value, run_research_experiment
from midi2score.train_seq2seq import run_seq2seq_training_loop


# =========================
# CLI 参数
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run seq2seq training directly or as a managed experiment."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/seq2seq_baseline.yaml"),
        help="Path to base YAML config",
    )

    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment id (if set → managed experiment mode)",
    )

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in dotted form, e.g. model.d_model=256",
    )

    parser.add_argument(
        "--note",
        default=None,
        help="Optional note stored in summary",
    )

    parser.add_argument(
        "--reference-best-loss",
        type=float,
        default=None,
        help="Optional reference validation loss",
    )

    parser.add_argument(
        "--allow-dirty-git",
        action="store_true",
        help="Allow running with dirty git",
    )

    return parser


# =========================
# 主函数
# =========================
def main() -> None:
    args = build_parser().parse_args()
    overrides = _parse_overrides(args.overrides)

    # =========================
    # 🟢 普通训练模式
    # =========================
    if args.experiment_id is None:
        if overrides or args.note is not None or args.reference_best_loss is not None:
            raise ValueError("--set/--note require --experiment-id")

        project_config = load_seq2seq_config(args.config)

        result = run_seq2seq_training_loop(
            project_config.model,
            project_config.data,
            project_config.training,
        )

        print(f"finished {len(result.losses)} training steps on {result.device}")
        print(
            f"final_step={result.final_step} elapsed_seconds={result.elapsed_seconds:.2f} "
            f"stopped_due_to_time_budget={result.stopped_due_to_time_budget}"
        )

        if result.best_validation_loss is not None:
            print(f"best validation loss {result.best_validation_loss:.4f}")

        if result.checkpoint_path is not None:
            print(f"saved checkpoint to {result.checkpoint_path}")

        if result.best_checkpoint_path is not None:
            print(f"saved best checkpoint to {result.best_checkpoint_path}")

        return

    # =========================
    # 🔵 Managed Experiment 模式
    # =========================
    summary = run_research_experiment(
        base_config_path=args.config,
        experiment_id=args.experiment_id,
        overrides=overrides,
        note=args.note,
        reference_best_validation_loss=args.reference_best_loss,
        require_clean_git=not args.allow_dirty_git,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


# =========================
# override 解析
# =========================
def _parse_overrides(pairs: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be KEY=VALUE, got {pair}")

        key, raw_value = pair.split("=", maxsplit=1)

        if not key:
            raise ValueError(f"Override key empty: {pair}")

        overrides[key] = parse_override_value(raw_value)

    return overrides


if __name__ == "__main__":
    main()