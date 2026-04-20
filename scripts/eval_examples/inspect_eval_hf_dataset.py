from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_from_disk

# =========================
# Global Variables
# =========================
EVAL_ROOT = Path("DATA/eval_dataset")
HF_DATASET_PATH = EVAL_ROOT / "hf_dataset"
SPLIT = "test"
PREVIEW_ROWS = 3


def summarize_paths(row: dict, eval_root: Path) -> dict[str, bool]:
    xml_rel = row.get("truncated_musicxml_path")
    midi_rel = row.get("truncated_midi_path")

    return {
        "xml_path_exists": bool(xml_rel) and (eval_root / xml_rel).exists(),
        "midi_path_exists": bool(midi_rel) and (eval_root / midi_rel).exists(),
    }


def main() -> None:
    if not HF_DATASET_PATH.exists():
        print("HF dataset path does not exist.")
        print(f"Current HF_DATASET_PATH: {HF_DATASET_PATH.resolve()}")
        print("Please update EVAL_ROOT / HF_DATASET_PATH in this script.")
        return

    ds = load_from_disk(str(HF_DATASET_PATH))
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got: {type(ds)}")

    if SPLIT not in ds:
        raise KeyError(f"Split {SPLIT!r} not found. Available: {list(ds.keys())}")

    split_ds = ds[SPLIT]

    print("=== HF Evaluation Dataset Quick Inspect ===")
    print(f"eval_root           : {EVAL_ROOT.resolve()}")
    print(f"hf_dataset_path     : {HF_DATASET_PATH.resolve()}")
    print(f"available_splits    : {list(ds.keys())}")
    print(f"active_split        : {SPLIT}")
    print(f"num_rows            : {len(split_ds)}")
    print(f"num_columns         : {len(split_ds.column_names)}")
    print(f"column_names        : {split_ds.column_names}")

    if len(split_ds) == 0:
        print("Split is empty. Nothing to preview.")
        return

    if "selected_variant" in split_ds.column_names:
        variant_counts: dict[str, int] = {}
        for value in split_ds["selected_variant"]:
            variant_counts[value] = variant_counts.get(value, 0) + 1
        print(f"selected_variant_counts: {variant_counts}")

    n_preview = min(PREVIEW_ROWS, len(split_ds))
    print(f"\n=== Preview first {n_preview} rows ===")

    for i in range(n_preview):
        row = split_ds[i]
        path_status = summarize_paths(row, EVAL_ROOT)

        print(f"\n--- row {i} ---")
        print(f"id                    : {row.get('id')}")
        print(f"selected_variant      : {row.get('selected_variant')}")
        print(f"selected_source_length: {row.get('selected_source_length')}")
        print(f"selected_target_length: {row.get('selected_target_length')}")
        print(f"xml_path              : {row.get('truncated_musicxml_path')}")
        print(f"midi_path             : {row.get('truncated_midi_path')}")
        print(f"xml_conversion_ok     : {row.get('xml_conversion_ok')}")
        print(f"midi_conversion_ok    : {row.get('midi_conversion_ok')}")
        print(f"path_exists           : {path_status}")


if __name__ == "__main__":
    main()
