from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

# Allow `python hf_dataset/hf_dataset_eval_generate.py ...` to resolve project imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.cpword_tokenizer_config import cpword_tokenizer
from tokenizer.musicxml_tokenizer import MusicXMLTokenizer

NOISE_FIELDS = {
    "clean": "midi_clean_ids",
    "light": "midi_light_ids",
    "heavy": "midi_heavy_ids",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a test-only evaluation dataset from structure-aware truncated seq2seq data. "
            "Outputs a HuggingFace dataset plus truncated .xml/.mid artifacts."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Path to truncated DatasetDict")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root folder")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="LMX BPE tokenizer JSON path")
    parser.add_argument("--split", type=str, default="test", help="Split to export (default: test)")
    parser.add_argument("--ratio-clean", type=float, default=1.0, help="Variant assignment weight for clean")
    parser.add_argument("--ratio-light", type=float, default=1.0, help="Variant assignment weight for light")
    parser.add_argument("--ratio-heavy", type=float, default=1.0, help="Variant assignment weight for heavy")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by deterministic variant assignment")
    parser.add_argument(
        "--id-style",
        type=str,
        choices=["pred_seq2seq", "plain"],
        default="pred_seq2seq",
        help=(
            "ID format. 'pred_seq2seq' generates sample_{i}_0 so it matches current pred_seq2seq fallback names."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument(
        "--strict-midi-decode",
        action="store_true",
        help="Fail fast when CPWord -> MIDI decode fails for any sample",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output root if it exists")
    return parser


def normalize_ratios(clean: float, light: float, heavy: float) -> dict[str, float]:
    ratios = {"clean": clean, "light": light, "heavy": heavy}
    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"--ratio-{name} must be >= 0")

    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one ratio among clean/light/heavy must be > 0")

    return {name: value / total for name, value in ratios.items()}


def make_id(index: int, *, id_style: str) -> str:
    if id_style == "pred_seq2seq":
        return f"sample_{index}_0"
    return f"eval_{index:07d}"


def stable_random_unit(sample_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


def assign_variant(sample_id: str, *, ratios: dict[str, float], seed: int) -> str:
    x = stable_random_unit(sample_id, seed)
    c1 = ratios["clean"]
    c2 = ratios["clean"] + ratios["light"]

    if x < c1:
        return "clean"
    if x < c2:
        return "light"
    return "heavy"


def clamp_cutoff(cutoff: int, length: int) -> int:
    if cutoff < 0:
        return 0
    if cutoff > length:
        return length
    return cutoff


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_midi(score: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(score, "dump_midi"):
        score.dump_midi(path)
        return
    if hasattr(score, "write_midi"):
        score.write_midi(path)
        return

    raise RuntimeError("Decoded MIDI object has no dump_midi/write_midi method")


def decode_cpword_to_midi(cpword_ids: list[list[int]]) -> Any:
    # miditok CPWord decode returns a symusic.Score object in this environment.
    return cpword_tokenizer.decode(cpword_ids)


def decode_bpe_to_lmx_text(token_ids: list[int], tokenizer: MusicXMLTokenizer) -> str:
    if not token_ids:
        return ""

    byte_str = tokenizer.bpe.decode(token_ids)
    lmx_text = tokenizer.mapper.decode_to_lmx(byte_str)
    return lmx_text.strip()


def main() -> None:
    args = build_parser().parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    input_path = args.input_path
    output_root = args.output_root

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    ratios = normalize_ratios(args.ratio_clean, args.ratio_light, args.ratio_heavy)

    print(f"=== Loading truncated dataset from {input_path} ===")
    dataset_dict = load_from_disk(str(input_path))
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError("Input path must point to a Hugging Face DatasetDict")

    if args.split not in dataset_dict:
        raise ValueError(f"Split {args.split!r} is missing. Available splits: {list(dataset_dict.keys())}")

    split_dataset = dataset_dict[args.split]
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
        "truncated_clean",
        "truncated_light",
        "truncated_heavy",
        "measure_count_clean",
        "measure_count_light",
        "measure_count_heavy",
        "boundary_aligned_clean",
        "boundary_aligned_light",
        "boundary_aligned_heavy",
    }
    missing = sorted(required_columns.difference(split_dataset.column_names))
    if missing:
        raise ValueError(f"Split {args.split!r} missing required columns: {missing}")

    if args.max_samples is not None:
        n = min(len(split_dataset), args.max_samples)
        split_dataset = split_dataset.select(range(n))

    out_hf_path = output_root / "hf_dataset"
    out_xml_dir = output_root / "musicxml"
    out_lmx_dir = output_root / "lmx"
    out_midi_dir = output_root / "midi"
    out_manifest_dir = output_root / "manifests"
    out_logs_dir = output_root / "logs"

    out_xml_dir.mkdir(parents=True, exist_ok=True)
    out_lmx_dir.mkdir(parents=True, exist_ok=True)
    out_midi_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading tokenizer for LMX -> MusicXML decode ===")
    tokenizer = MusicXMLTokenizer()
    tokenizer.load_bpe_model(str(args.tokenizer_path))
    tokenizer.eval_mode()

    rows: list[dict[str, Any]] = []
    variant_counter: Counter[str] = Counter()
    xml_success = 0
    lmx_success = 0
    midi_success = 0
    xml_failures: list[dict[str, str]] = []
    lmx_failures: list[dict[str, str]] = []
    midi_failures: list[dict[str, str]] = []

    print(f"=== Generating evaluation assets for split '{args.split}' ({len(split_dataset)} samples) ===")
    for index in tqdm(range(len(split_dataset)), desc="Generating eval dataset"):
        row = split_dataset[index]
        sample_id = make_id(index, id_style=args.id_style)

        selected_variant = assign_variant(sample_id, ratios=ratios, seed=args.seed)
        variant_counter[selected_variant] += 1

        midi_field = NOISE_FIELDS[selected_variant]
        selected_cpword_ids = row[midi_field]

        lmx_ids = row["lmx_ids"]
        cutoff = clamp_cutoff(int(row[f"lmx_cutoff_{selected_variant}"]), len(lmx_ids))
        selected_lmx_ids = lmx_ids[:cutoff]

        xml_rel = Path("musicxml") / f"{sample_id}.xml"
        lmx_rel = Path("lmx") / f"{sample_id}.lmx"
        midi_rel = Path("midi") / f"{sample_id}.mid"

        xml_ok = False
        lmx_ok = False
        midi_ok = False
        xml_error = ""
        lmx_error = ""
        midi_error = ""

        try:
            lmx_text = decode_bpe_to_lmx_text(selected_lmx_ids, tokenizer)
            save_text(output_root / lmx_rel, lmx_text)
            lmx_ok = True
            lmx_success += 1
        except Exception as exc:  # noqa: BLE001
            lmx_error = str(exc)
            lmx_failures.append({"id": sample_id, "error": lmx_error})

        try:
            xml_text = tokenizer.decode_bpe_to_musicxml(selected_lmx_ids)
            save_text(output_root / xml_rel, xml_text)
            xml_ok = True
            xml_success += 1
        except Exception as exc:  # noqa: BLE001
            xml_error = str(exc)
            xml_failures.append({"id": sample_id, "error": xml_error})

        try:
            midi_score = decode_cpword_to_midi(selected_cpword_ids)
            save_midi(midi_score, output_root / midi_rel)
            midi_ok = True
            midi_success += 1
        except Exception as exc:  # noqa: BLE001
            midi_error = str(exc)
            midi_failures.append({"id": sample_id, "error": midi_error})
            if args.strict_midi_decode:
                raise RuntimeError(f"Failed CPWord -> MIDI decode for {sample_id}: {exc}") from exc

        new_row = dict(row)
        new_row.update(
            {
                "id": sample_id,
                "selected_variant": selected_variant,
                "selected_cpword_ids": selected_cpword_ids,
                "selected_lmx_ids": selected_lmx_ids,
                "selected_source_length": len(selected_cpword_ids),
                "selected_target_length": len(selected_lmx_ids),
                "truncated_musicxml_path": str(xml_rel),
                "truncated_lmx_path": str(lmx_rel),
                "truncated_midi_path": str(midi_rel),
                "xml_conversion_ok": xml_ok,
                "lmx_conversion_ok": lmx_ok,
                "midi_conversion_ok": midi_ok,
                "xml_conversion_error": xml_error,
                "lmx_conversion_error": lmx_error,
                "midi_conversion_error": midi_error,
            }
        )
        rows.append(new_row)

    eval_dataset = Dataset.from_list(rows)
    eval_dataset_dict = DatasetDict({args.split: eval_dataset})

    print(f"=== Saving evaluation HuggingFace dataset to {out_hf_path} ===")
    out_hf_path.parent.mkdir(parents=True, exist_ok=True)
    eval_dataset_dict.save_to_disk(str(out_hf_path))

    manifest_path = out_manifest_dir / f"{args.split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            manifest_row = {
                "id": row["id"],
                "selected_variant": row["selected_variant"],
                "selected_source_length": row["selected_source_length"],
                "selected_target_length": row["selected_target_length"],
                "truncated_musicxml_path": row["truncated_musicxml_path"],
                "truncated_lmx_path": row["truncated_lmx_path"],
                "truncated_midi_path": row["truncated_midi_path"],
                "xml_conversion_ok": row["xml_conversion_ok"],
                "lmx_conversion_ok": row["lmx_conversion_ok"],
                "midi_conversion_ok": row["midi_conversion_ok"],
                "xml_conversion_error": row["xml_conversion_error"],
                "lmx_conversion_error": row["lmx_conversion_error"],
                "midi_conversion_error": row["midi_conversion_error"],
            }
            f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    xml_failure_path = out_logs_dir / "xml_failures.json"
    lmx_failure_path = out_logs_dir / "lmx_failures.json"
    midi_failure_path = out_logs_dir / "midi_failures.json"
    xml_failure_path.write_text(json.dumps(xml_failures, ensure_ascii=False, indent=2), encoding="utf-8")
    lmx_failure_path.write_text(json.dumps(lmx_failures, ensure_ascii=False, indent=2), encoding="utf-8")
    midi_failure_path.write_text(json.dumps(midi_failures, ensure_ascii=False, indent=2), encoding="utf-8")

    n_samples = len(rows)
    summary = {
        "input_path": str(input_path),
        "output_root": str(output_root),
        "split": args.split,
        "num_samples": n_samples,
        "id_style": args.id_style,
        "seed": args.seed,
        "requested_ratios": {
            "clean": args.ratio_clean,
            "light": args.ratio_light,
            "heavy": args.ratio_heavy,
        },
        "normalized_ratios": ratios,
        "assigned_variant_counts": dict(variant_counter),
        "assigned_variant_rates": {
            key: (value / n_samples if n_samples > 0 else 0.0)
            for key, value in variant_counter.items()
        },
        "xml_conversion_success": xml_success,
        "lmx_conversion_success": lmx_success,
        "midi_conversion_success": midi_success,
        "xml_conversion_success_rate": (xml_success / n_samples if n_samples > 0 else 0.0),
        "lmx_conversion_success_rate": (lmx_success / n_samples if n_samples > 0 else 0.0),
        "midi_conversion_success_rate": (midi_success / n_samples if n_samples > 0 else 0.0),
        "manifest_path": str(manifest_path),
        "hf_dataset_path": str(out_hf_path),
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print("=== Done ===")
    print(f"Saved HF dataset : {out_hf_path}")
    print(f"Saved MusicXML   : {out_xml_dir}")
    print(f"Saved LMX        : {out_lmx_dir}")
    print(f"Saved MIDI       : {out_midi_dir}")
    print(f"Saved manifest   : {manifest_path}")
    print(f"Saved summary    : {summary_path}")


if __name__ == "__main__":
    main()
