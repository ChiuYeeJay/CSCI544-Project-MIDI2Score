from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import tqdm
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

sys.path.append(".")
from tokenizer.musicxml_tokenizer import MusicXMLTokenizer

NOISE_FIELDS = {
    "clean": "midi_clean_ids",
    "light": "midi_light_ids",
    "heavy": "midi_heavy_ids",
}


def infer_cpword_dim(*midi_variants: list[list[int]]) -> int:
    for midi_ids in midi_variants:
        if midi_ids and isinstance(midi_ids[0], (list, tuple)) and len(midi_ids[0]) > 0:
            return len(midi_ids[0])
    return 7


def ensure_non_empty_midi(midi_ids: list[list[int]], cpword_dim: int) -> list[list[int]]:
    if midi_ids:
        return midi_ids
    return [[0] * cpword_dim]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a structure-aware truncated seq2seq Hugging Face dataset from an existing tokenized dataset."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input tokenized DatasetDict path")
    parser.add_argument("--output-path", type=Path, required=True, help="Output truncated DatasetDict path")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Tokenizer JSON path")
    parser.add_argument("--max-source-length", type=int, required=True, help="Max CPWord length")
    parser.add_argument("--max-target-length", type=int, required=True, help="Max LMX BPE length")
    parser.add_argument("--lookahead-tokens", type=int, default=0, help="Tokens allowed after measure boundary")
    parser.add_argument("--lookahead-clean", type=int, default=None, help="Override lookahead for clean variant (default: 0)")
    parser.add_argument("--lookahead-light", type=int, default=None, help="Override lookahead for light variant")
    parser.add_argument("--lookahead-heavy", type=int, default=None, help="Override lookahead for heavy variant")
    parser.add_argument("--batch-size", type=int, default=1024, help="Dataset map batch size")
    parser.add_argument("--num-proc", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="dataset.map workers")
    parser.add_argument("--stats-path", type=Path, default=None, help="Optional JSON file for split stats")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    return parser


def resolve_measure_start_ids(tokenizer_path: Path) -> set[int]:
    tokenizer = MusicXMLTokenizer()
    tokenizer.load_bpe_model(str(tokenizer_path))

    measure_byte = tokenizer.mapper.get_token_byte("measure")
    if not measure_byte:
        raise RuntimeError("Could not resolve byte mapping for LMX 'measure' token.")

    vocab = tokenizer.bpe.bpe_tokenizer.get_vocab()
    ids = {token_id for token, token_id in vocab.items() if token.startswith(measure_byte)}

    if not ids:
        ids = {token_id for token, token_id in vocab.items() if measure_byte in token}

    if not ids:
        raise RuntimeError("No BPE token IDs matched the LMX measure byte in tokenizer vocab.")

    return ids


def lmx_measure_starts(lmx_ids: list[int], lmx_measure_start_ids: set[int]) -> list[int]:
    starts = [idx for idx, token_id in enumerate(lmx_ids) if token_id in lmx_measure_start_ids]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    return sorted(set(starts))


def cpword_measure_starts(midi_ids: list[list[int]]) -> list[int]:
    starts: list[int] = []
    for idx, token in enumerate(midi_ids):
        if isinstance(token, (list, tuple)) and len(token) > 1 and int(token[1]) == 2:
            starts.append(idx)

    if not starts or starts[0] != 0:
        starts = [0] + starts
    return sorted(set(starts))


def prefix_end_from_measure_count(starts: list[int], seq_len: int, measure_count: int) -> int:
    if measure_count <= 0:
        return 0
    if measure_count >= len(starts):
        return seq_len
    return starts[measure_count]


def truncate_pair_by_measure(
    lmx_ids: list[int],
    midi_ids: list[list[int]],
    *,
    lmx_measure_start_ids: set[int],
    max_source_length: int,
    max_target_length: int,
    lookahead_tokens: int,
) -> dict[str, int | bool]:
    lmx_starts = lmx_measure_starts(lmx_ids, lmx_measure_start_ids)
    midi_starts = cpword_measure_starts(midi_ids)

    detected_lmx_boundaries = any(pos > 0 for pos in lmx_starts)
    detected_midi_boundaries = any(pos > 0 for pos in midi_starts)

    max_shared_measures = min(len(lmx_starts), len(midi_starts))
    best: tuple[int, int, int, int, int] | None = None

    for measure_count in range(1, max_shared_measures + 1):
        lmx_base_end = prefix_end_from_measure_count(lmx_starts, len(lmx_ids), measure_count)
        midi_base_end = prefix_end_from_measure_count(midi_starts, len(midi_ids), measure_count)

        lmx_end = min(len(lmx_ids), lmx_base_end + lookahead_tokens)
        midi_end = min(len(midi_ids), midi_base_end + lookahead_tokens)

        if lmx_end <= max_target_length and midi_end <= max_source_length:
            best = (measure_count, lmx_end, midi_end, lmx_base_end, midi_base_end)
        else:
            break

    if best is None:
        lmx_end = min(len(lmx_ids), max_target_length)
        midi_end = min(len(midi_ids), max_source_length)
        return {
            "measure_count": 0,
            "lmx_end": lmx_end,
            "midi_end": midi_end,
            "boundary_aligned": False,
            "lookahead_lmx": 0,
            "lookahead_midi": 0,
            "truncated": lmx_end < len(lmx_ids) or midi_end < len(midi_ids),
        }

    measure_count, lmx_end, midi_end, lmx_base_end, midi_base_end = best
    return {
        "measure_count": measure_count,
        "lmx_end": lmx_end,
        "midi_end": midi_end,
        "boundary_aligned": detected_lmx_boundaries and detected_midi_boundaries,
        "lookahead_lmx": max(0, lmx_end - lmx_base_end),
        "lookahead_midi": max(0, midi_end - midi_base_end),
        "truncated": lmx_end < len(lmx_ids) or midi_end < len(midi_ids),
    }


def process_batch(
    examples: dict,
    *,
    lmx_measure_start_ids: set[int],
    max_source_length: int,
    max_target_length: int,
    lookahead_tokens_by_variant: dict[str, int],
) -> dict:
    output = {
        "lmx_ids": [],
        "lmx_length": [],
    }

    for variant, midi_field in NOISE_FIELDS.items():
        output[midi_field] = []
        output[f"source_length_{variant}"] = []
        output[f"target_length_{variant}"] = []
        output[f"lmx_cutoff_{variant}"] = []
        output[f"measure_count_{variant}"] = []
        output[f"boundary_aligned_{variant}"] = []
        output[f"lookahead_lmx_{variant}"] = []
        output[f"lookahead_midi_{variant}"] = []
        output[f"truncated_{variant}"] = []

    n_examples = len(examples["lmx_ids"])
    for row_idx in range(n_examples):
        lmx_ids = examples["lmx_ids"][row_idx]
        cpword_dim = infer_cpword_dim(
            examples["midi_clean_ids"][row_idx],
            examples["midi_light_ids"][row_idx],
            examples["midi_heavy_ids"][row_idx],
        )

        per_variant: dict[str, dict] = {}
        for variant, midi_field in NOISE_FIELDS.items():
            midi_ids = examples[midi_field][row_idx]
            lookahead_tokens = lookahead_tokens_by_variant[variant]
            per_variant[variant] = truncate_pair_by_measure(
                lmx_ids,
                midi_ids,
                lmx_measure_start_ids=lmx_measure_start_ids,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                lookahead_tokens=lookahead_tokens,
            )

        global_lmx_end = max(int(result["lmx_end"]) for result in per_variant.values())
        lmx_out = lmx_ids[:global_lmx_end]

        output["lmx_ids"].append(lmx_out)
        output["lmx_length"].append(len(lmx_out))

        for variant, midi_field in NOISE_FIELDS.items():
            result = per_variant[variant]
            midi_end = int(result["midi_end"])
            lmx_end = int(result["lmx_end"])
            midi_out = examples[midi_field][row_idx][:midi_end]
            midi_out = ensure_non_empty_midi(midi_out, cpword_dim)

            output[midi_field].append(midi_out)
            output[f"source_length_{variant}"].append(len(midi_out))
            output[f"target_length_{variant}"].append(lmx_end)
            output[f"lmx_cutoff_{variant}"].append(lmx_end)
            output[f"measure_count_{variant}"].append(int(result["measure_count"]))
            output[f"boundary_aligned_{variant}"].append(bool(result["boundary_aligned"]))
            output[f"lookahead_lmx_{variant}"].append(int(result["lookahead_lmx"]))
            output[f"lookahead_midi_{variant}"].append(int(result["lookahead_midi"]))
            output[f"truncated_{variant}"].append(bool(result["truncated"]))

    return output


def collect_split_stats(dataset: Dataset) -> dict[str, float | int]:
    n_samples = len(dataset)
    stats: dict[str, float | int] = {"num_samples": n_samples}

    if n_samples == 0:
        return stats

    batch_size = 4096
    for variant in NOISE_FIELDS:
        source_sum = 0.0
        target_sum = 0.0
        measure_sum = 0.0
        aligned_sum = 0.0
        truncated_sum = 0.0

        for batch in tqdm.tqdm(
            dataset.iter(batch_size=batch_size), 
            total=math.ceil(n_samples / batch_size), 
            desc=f"Collecting stats for variant '{variant}'"
        ):
            source_vals = batch[f"source_length_{variant}"]
            target_vals = batch[f"target_length_{variant}"]
            measure_vals = batch[f"measure_count_{variant}"]
            aligned_vals = batch[f"boundary_aligned_{variant}"]
            truncated_vals = batch[f"truncated_{variant}"]

            source_sum += float(sum(source_vals))
            target_sum += float(sum(target_vals))
            measure_sum += float(sum(measure_vals))
            aligned_sum += float(sum(bool(v) for v in aligned_vals))
            truncated_sum += float(sum(bool(v) for v in truncated_vals))

        stats[f"avg_source_length_{variant}"] = source_sum / n_samples
        stats[f"avg_target_length_{variant}"] = target_sum / n_samples
        stats[f"avg_measure_count_{variant}"] = measure_sum / n_samples
        stats[f"boundary_aligned_rate_{variant}"] = aligned_sum / n_samples
        stats[f"truncated_rate_{variant}"] = truncated_sum / n_samples

    return stats


def main() -> None:
    args = build_parser().parse_args()

    if args.max_source_length <= 1:
        raise ValueError("--max-source-length must be >= 2")
    if args.max_target_length <= 1:
        raise ValueError("--max-target-length must be >= 2")
    if args.lookahead_tokens < 0:
        raise ValueError("--lookahead-tokens must be >= 0")
    for arg_name in ("lookahead_clean", "lookahead_light", "lookahead_heavy"):
        value = getattr(args, arg_name)
        if value is not None and value < 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be positive")

    input_path = args.input_path
    output_path = args.output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}. Use --overwrite to replace it.")

    if output_path.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_path)

    print("=== Resolving LMX measure-start BPE token IDs ===")
    lmx_measure_start_ids = resolve_measure_start_ids(args.tokenizer_path)
    print(f"Detected {len(lmx_measure_start_ids)} measure-start BPE IDs")

    lookahead_by_variant = {
        "clean": 0 if args.lookahead_clean is None else args.lookahead_clean,
        "light": args.lookahead_tokens if args.lookahead_light is None else args.lookahead_light,
        "heavy": args.lookahead_tokens if args.lookahead_heavy is None else args.lookahead_heavy,
    }
    print(
        "Lookahead per variant -> "
        f"clean: {lookahead_by_variant['clean']}, "
        f"light: {lookahead_by_variant['light']}, "
        f"heavy: {lookahead_by_variant['heavy']}"
    )

    print(f"=== Loading input dataset from {input_path} ===")
    dataset_dict = load_from_disk(str(input_path))
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError("Input path must point to a Hugging Face DatasetDict")

    processed_splits: dict[str, Dataset] = {}
    stats_payload: dict[str, dict[str, float | int]] = {}

    for split_name, split_dataset in dataset_dict.items():
        print(f"\n--- Processing split: {split_name} ({len(split_dataset)} samples) ---")

        required_columns = {"lmx_ids", *NOISE_FIELDS.values()}
        missing = [col for col in required_columns if col not in split_dataset.column_names]
        if missing:
            raise ValueError(f"Split '{split_name}' missing required columns: {missing}")

        processed_split = split_dataset.map(
            process_batch,
            fn_kwargs={
                "lmx_measure_start_ids": lmx_measure_start_ids,
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "lookahead_tokens_by_variant": lookahead_by_variant,
            },
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=split_dataset.column_names,
            desc=f"Structure-aware truncation ({split_name})",
        )

        processed_splits[split_name] = processed_split
        stats_payload[split_name] = collect_split_stats(processed_split)

    output_dataset = DatasetDict(processed_splits)

    print(f"\n=== Saving truncated dataset to {output_path} ===")
    os.makedirs(output_path, exist_ok=True)
    output_dataset.save_to_disk(str(output_path))

    stats_path = args.stats_path or (output_path / "truncation_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "tokenizer_path": str(args.tokenizer_path),
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "lookahead_tokens": args.lookahead_tokens,
                "lookahead_clean": lookahead_by_variant["clean"],
                "lookahead_light": lookahead_by_variant["light"],
                "lookahead_heavy": lookahead_by_variant["heavy"],
                "splits": stats_payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Saved stats to {stats_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
