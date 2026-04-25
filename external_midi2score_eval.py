from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# ====== paths ======
PROJECT_ROOT = Path(__file__).resolve().parent
EXTERNAL_REPO_ROOT = Path(r"D:\csci544 project\MIDI2ScoreTransformer-main")
EXTERNAL_PKG_ROOT = EXTERNAL_REPO_ROOT / "midi2scoretransformer"

if str(EXTERNAL_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_PKG_ROOT))

# local evaluator
from evaluation import evaluate_xml_dirs

# external model code
from tokenizer import MultistreamTokenizer
from models.roformer import Roformer
from score_utils import postprocess_score


device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def infer(x, model, overlap=64, chunk=512, verbose=True, kv_cache=True, top_k=1):
    single_example = x["pitch"].ndim == 2
    if single_example:
        x = {k: v.unsqueeze(0) for k, v in x.items()}

    x = {k: v.to(model.device) for k, v in x.items()}

    if chunk <= overlap:
        raise ValueError("chunk must be greater than overlap.")

    y_full = None

    for i in range(0, max(x["pitch"].shape[1] - overlap, 1), chunk - overlap):
        if verbose:
            print(f"Infer {i} / {x['pitch'].shape[1]}", end="\r")

        x_chunk = {k: v[:, i:i + chunk] for k, v in x.items()}

        if i == 0 or overlap == 0:
            y_hat = model.generate(
                x=x_chunk,
                top_k=top_k,
                max_length=chunk,
                kv_cache=kv_cache,
            )
        else:
            y_hat_prev = {
                k: v[:, -overlap:] if k != "pad" else v[:, -overlap:, 0]
                for k, v in y_full.items()
            }
            y_hat = model.generate(
                x=x_chunk,
                y=y_hat_prev,
                top_k=top_k,
                max_length=chunk,
                kv_cache=kv_cache,
            )
            y_hat = {k: v[:, overlap:] for k, v in y_hat.items()}

        if y_full is None:
            y_full = y_hat
        else:
            for k in y_full:
                y_full[k] = torch.cat((y_full[k], y_hat[k]), dim=1)

    if single_example:
        y_full = {k: v[0].cpu() for k, v in y_full.items()}
    else:
        y_full = {k: v.cpu() for k, v in y_full.items()}

    return y_full


def resolve_path(base: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def load_manifest_rows(manifest_path: Path, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_metrics_block(title: str, grouped_summary: dict, validity: dict | None = None) -> list[str]:
    lines = [title]
    for group in ["overall", "clean", "light", "heavy"]:
        if group not in grouped_summary:
            continue

        summary = grouped_summary[group]
        available = None
        evaluated = None
        if validity and "group_validity" in validity and group in validity["group_validity"]:
            available = validity["group_validity"][group].get("available")
            evaluated = validity["group_validity"][group].get("evaluated")

        if available is not None and evaluated is not None:
            lines.append(f"[{group}] available={available} evaluated={evaluated}")
        else:
            lines.append(f"[{group}]")

        if not summary:
            lines.append("  summary: <empty>")
            continue

        for k, v in summary.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")
    return lines


def build_group_validity(selected_rows: list[dict], eval_payload: dict) -> dict:
    variant_counter = Counter()
    for row in selected_rows:
        variant = str(row.get("selected_variant", "unknown"))
        variant_counter[variant] += 1

    grouped_summary = eval_payload.get("grouped_summary", {})
    group_validity = {
        "overall": {
            "available": len(selected_rows),
            "evaluated": grouped_summary.get("overall", {}).get("num_files", 0),
        }
    }
    for group in ["clean", "light", "heavy"]:
        group_validity[group] = {
            "available": int(variant_counter.get(group, 0)),
            "evaluated": grouped_summary.get(group, {}).get("num_files", 0),
        }

    return {
        "variant_counts": dict(variant_counter),
        "group_validity": group_validity,
    }


def build_gt_subset(sample_ids: list[str], gt_xml_dir: Path, gt_subset_dir: Path) -> None:
    gt_subset_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        src = gt_xml_dir / f"{sample_id}.xml"
        dst = gt_subset_dir / f"{sample_id}.xml"
        if src.exists():
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Run external MIDI2ScoreTransformer on the first N eval MIDI files and evaluate with local evaluation.py"
    )
    parser.add_argument("--eval-root", type=str, default="DATA/eval_dataset")
    parser.add_argument("--manifest-jsonl", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=r"D:\csci544 project\MIDI2ScoreTransformer-main\MIDI2ScoreTF.ckpt")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0, help="0-based index into the selected sample list.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose output XML already exists.")
    parser.add_argument("--onset-tol", type=float, default=0.0)
    parser.add_argument("--duration-tol", type=float, default=0.0)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--skip-ids", nargs="*", default=[], help="Sample ids to skip explicitly, e.g. sample_346_0")
    args = parser.parse_args()

    eval_root = resolve_path(PROJECT_ROOT, args.eval_root)
    out_dir = resolve_path(PROJECT_ROOT, args.out_dir)
    ckpt_path = Path(args.ckpt).resolve() if Path(args.ckpt).is_absolute() else (EXTERNAL_REPO_ROOT / args.ckpt).resolve()

    manifest_path = (
        Path(args.manifest_jsonl).resolve()
        if args.manifest_jsonl
        else (eval_root / "manifests" / "test.jsonl")
    )

    midi_dir = eval_root / "midi"
    gt_xml_dir = eval_root / "musicxml"
    pred_xml_dir = out_dir / "musicxml"
    gt_subset_dir = out_dir / "gt_subset"

    pred_xml_dir.mkdir(parents=True, exist_ok=True)
    gt_subset_dir.mkdir(parents=True, exist_ok=True)

    print("Loading external model...")
    model = Roformer.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.to(device)
    model.eval()

    selected_rows = load_manifest_rows(manifest_path, args.max_samples)
    if args.start_index < 0 or args.start_index >= len(selected_rows):
        raise ValueError(f"--start-index {args.start_index} out of range for {len(selected_rows)} selected samples")

    sample_ids = [str(row["id"]) for row in selected_rows]
    skip_ids = set(args.skip_ids)
    print(f"Loaded {len(sample_ids)} sample ids from {manifest_path}")
    print(f"Running from start_index={args.start_index}, skip_existing={args.skip_existing}")

    build_gt_subset(sample_ids, gt_xml_dir, gt_subset_dir)

    generation_failures: list[dict] = []
    generated = 0
    skipped_existing = 0

    run_start_time = time.time()

    for idx in range(args.start_index, len(sample_ids)):
        sample_id = sample_ids[idx]
        midi_path = midi_dir / f"{sample_id}.mid"
        out_path = pred_xml_dir / f"{sample_id}.xml"

        print(f"\n[{idx + 1}/{len(sample_ids)}] {sample_id}")

        if sample_id in skip_ids:
            print(f"[SKIP] sample id is in --skip-ids: {sample_id}")
            continue

        print(f"\n[{idx + 1}/{len(sample_ids)}] {sample_id}")

        if args.skip_existing and out_path.exists():
            skipped_existing += 1
            generated += 1
            print(f"[SKIP] existing output found: {out_path}")
            continue

        if not midi_path.exists():
            generation_failures.append({"id": sample_id, "error": f"Missing MIDI: {midi_path}"})
            print(f"[ERROR] Missing MIDI: {midi_path}")
            continue

        sample_t0 = time.time()

        try:
            print("[Stage] tokenizing MIDI...")
            x = MultistreamTokenizer.tokenize_midi(str(midi_path))

            print("[Stage] generating tokens...")
            with torch.no_grad():
                y_hat = infer(
                    x,
                    model,
                    overlap=args.overlap,
                    chunk=args.chunk,
                    verbose=True,
                    kv_cache=args.kv_cache,
                    top_k=args.top_k,
                )

            print("\n[Stage] detokenizing to score...")
            score = MultistreamTokenizer.detokenize_mxl(y_hat)

            print("[Stage] postprocessing score...")
            score = postprocess_score(score)

            print("[Stage] writing MusicXML...")
            score.write("musicxml", fp=str(out_path))

            elapsed = time.time() - sample_t0
            generated += 1
            print(f"[OK] wrote {out_path}")
            print(f"[Time] sample elapsed: {elapsed:.2f}s")

        except Exception as exc:
            elapsed = time.time() - sample_t0
            generation_failures.append({"id": sample_id, "error": str(exc)})
            print(f"[ERROR] {sample_id}: {exc}")
            print(f"[Time] failed sample elapsed: {elapsed:.2f}s")
            continue

    total_elapsed = time.time() - run_start_time

    print("\nRunning local evaluation.py ...")
    eval_payload = evaluate_xml_dirs(
        pred_dir=str(pred_xml_dir),
        gt_dir=str(gt_subset_dir),
        manifest_jsonl=str(manifest_path),
        onset_tol=args.onset_tol,
        duration_tol=args.duration_tol,
    )

    validity_extra = build_group_validity(selected_rows, eval_payload)

    full_payload = {
        "config": {
            "eval_root": str(eval_root),
            "manifest_jsonl": str(manifest_path),
            "out_dir": str(out_dir),
            "ckpt": str(ckpt_path),
            "max_samples": args.max_samples,
            "start_index": args.start_index,
            "skip_existing": args.skip_existing,
            "skip_ids": sorted(skip_ids),
            "onset_tol": args.onset_tol,
            "duration_tol": args.duration_tol,
            "chunk": args.chunk,
            "overlap": args.overlap,
            "top_k": args.top_k,
            "kv_cache": args.kv_cache,
        },
        "generation": {
            "requested_samples": len(sample_ids),
            "variant_counts": validity_extra["variant_counts"],
            "generated_xml_files": generated,
            "skipped_existing": skipped_existing,
            "generation_failures": generation_failures,
            "total_elapsed_seconds": total_elapsed,
        },
        "xml": {
            **eval_payload,
            "group_validity": validity_extra["group_validity"],
        },
    }

    save_json(full_payload, out_dir / "evaluation.json")

    txt_lines = []
    txt_lines.append("================ External MIDI2ScoreTransformer Report ================")
    txt_lines.append("")
    txt_lines.append("[Run Configuration]")
    txt_lines.append(f"eval_root: {eval_root}")
    txt_lines.append(f"manifest_jsonl: {manifest_path}")
    txt_lines.append(f"out_dir: {out_dir}")
    txt_lines.append(f"checkpoint: {ckpt_path}")
    txt_lines.append(f"max_samples: {args.max_samples}")
    txt_lines.append(f"start_index: {args.start_index}")
    txt_lines.append(f"skip_existing: {args.skip_existing}")
    txt_lines.append(f"chunk: {args.chunk}")
    txt_lines.append(f"overlap: {args.overlap}")
    txt_lines.append(f"top_k: {args.top_k}")
    txt_lines.append(f"kv_cache: {args.kv_cache}")
    txt_lines.append(f"onset_tol: {args.onset_tol}")
    txt_lines.append(f"duration_tol: {args.duration_tol}")
    txt_lines.append("")

    txt_lines.append("[Generation Summary]")
    txt_lines.append(f"num_samples: {len(sample_ids)}")
    txt_lines.append(f"variant_counts: {validity_extra['variant_counts']}")
    txt_lines.append(f"generated_xml_files: {generated}")
    txt_lines.append(f"skipped_existing: {skipped_existing}")
    txt_lines.append(f"xml_generation_failures: {len(generation_failures)}")
    txt_lines.append(f"total_elapsed_seconds: {total_elapsed:.2f}")
    txt_lines.append("")

    txt_lines.append("[XML Validity]")
    for k, v in eval_payload["validity"].items():
        if isinstance(v, float):
            txt_lines.append(f"{k}: {v:.6f}")
        else:
            txt_lines.append(f"{k}: {v}")
    txt_lines.append("")

    grouped_summary = dict(eval_payload.get("grouped_summary", {}))
    if "overall" not in grouped_summary:
        grouped_summary["overall"] = eval_payload.get("summary", {})

    txt_lines.extend(
        format_metrics_block(
            "[XML Metrics]",
            grouped_summary=grouped_summary,
            validity={"group_validity": validity_extra["group_validity"]},
        )
    )
    txt_lines.append("")

    txt_lines.append("[Global Evaluation Failures]")
    txt_lines.append(f"xml_failures: {len(eval_payload.get('failures', []))}")

    report_text = "\n".join(txt_lines) + "\n"
    (out_dir / "evaluation.txt").write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"Saved evaluation json to: {out_dir / 'evaluation.json'}")
    print(f"Saved evaluation txt  to: {out_dir / 'evaluation.txt'}")


if __name__ == "__main__":
    main()