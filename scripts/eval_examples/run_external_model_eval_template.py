from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Template workflow for external models: read truncated MIDI IDs from the generated evaluation dataset, "
            "create prediction XML files, then run evaluation.py."
        )
    )
    parser.add_argument("--eval-root", type=Path, required=True, help="Evaluation dataset root")
    parser.add_argument(
        "--pred-xml-dir",
        type=Path,
        required=True,
        help="Directory where external model XML predictions are written",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Evaluation report path (default: <pred-xml-dir>/external_model_eval_report.json)",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable for subprocesses")
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate missing predictions using the placeholder function in this template",
    )
    return parser


def load_manifest(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate_prediction_xml_from_midi(midi_path: Path, output_xml_path: Path) -> None:
    """
    TODO: replace this placeholder with your external model inference implementation.

    Requirements:
    - input : one truncated MIDI file path
    - output: one MusicXML file path, named by the same sample id
    """
    raise NotImplementedError(
        "Please implement external model inference in generate_prediction_xml_from_midi()."
    )


def check_alignment(gt_xml_dir: Path, pred_xml_dir: Path) -> dict[str, object]:
    gt_files = sorted(p.name for p in gt_xml_dir.glob("*.xml"))
    pred_files = sorted(p.name for p in pred_xml_dir.glob("*.xml"))

    gt_set = set(gt_files)
    pred_set = set(pred_files)

    missing_in_pred = sorted(gt_set - pred_set)
    extra_in_pred = sorted(pred_set - gt_set)

    return {
        "num_gt_xml": len(gt_files),
        "num_pred_xml": len(pred_files),
        "missing_in_pred": missing_in_pred,
        "extra_in_pred": extra_in_pred,
    }


def run_command(command: list[str], *, cwd: Path) -> None:
    print("[CMD]", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = args.eval_root / "manifests" / "test.jsonl"
    gt_xml_dir = args.eval_root / "musicxml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not gt_xml_dir.exists():
        raise FileNotFoundError(f"Missing truncated MusicXML directory: {gt_xml_dir}")

    args.pred_xml_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.report_json or (args.pred_xml_dir / "external_model_eval_report.json")
    precheck_path = args.pred_xml_dir / "alignment_precheck.json"

    manifest_rows = load_manifest(manifest_path)

    if args.generate_missing:
        for row in manifest_rows:
            sample_id = row["id"]
            pred_xml_path = args.pred_xml_dir / f"{sample_id}.xml"
            if pred_xml_path.exists():
                continue

            midi_rel = row["truncated_midi_path"]
            midi_path = args.eval_root / midi_rel
            generate_prediction_xml_from_midi(midi_path=midi_path, output_xml_path=pred_xml_path)

    alignment = check_alignment(gt_xml_dir=gt_xml_dir, pred_xml_dir=args.pred_xml_dir)
    precheck_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_command = [
        args.python_bin,
        "evaluation.py",
        "--pred_xml_dir",
        str(args.pred_xml_dir),
        "--gt_xml_dir",
        str(gt_xml_dir),
        "--save_json",
        str(report_json),
    ]
    run_command(eval_command, cwd=project_root)

    print("=== Done ===")
    print(f"Manifest         : {manifest_path}")
    print(f"Ground-truth XML : {gt_xml_dir}")
    print(f"Predicted XML    : {args.pred_xml_dir}")
    print(f"Precheck log     : {precheck_path}")
    print(f"Evaluation report: {report_json}")


if __name__ == "__main__":
    main()
