from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Allow `python scripts/eval_examples/run_our_model_eval_example.py ...` to resolve project imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.musicxml_tokenizer import MusicXMLTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Example end-to-end workflow for our model: generate LMX predictions, convert to MusicXML, "
            "and run XML evaluation against the generated evaluation dataset."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True, help="Base seq2seq yaml config")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--eval-root", type=Path, required=True, help="Evaluation dataset root")
    parser.add_argument("--work-dir", type=Path, required=True, help="Working directory for outputs")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable for subprocesses")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Override tokenizer path")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples for prediction")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--top-k", type=int, default=1, help="Generation top-k")
    parser.add_argument("--skip-pred", action="store_true", help="Skip LMX prediction stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip final evaluation stage")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Evaluation report path (default: <work-dir>/our_model_eval_report.json)",
    )
    return parser


def save_temp_config_for_eval(base_config_path: Path, eval_hf_dataset_path: Path, output_path: Path, tokenizer_path: Path | None) -> Path:
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    if "data" not in cfg:
        raise ValueError("Config must contain a top-level 'data' section")

    cfg["data"]["dataset_path"] = str(eval_hf_dataset_path)
    if tokenizer_path is not None:
        cfg["data"]["tokenizer_path"] = str(tokenizer_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return output_path


def convert_lmx_dir_to_xml(pred_lmx_dir: Path, pred_xml_dir: Path, tokenizer: MusicXMLTokenizer) -> dict[str, object]:
    pred_xml_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    converted = 0

    lmx_files = sorted(pred_lmx_dir.glob("*.lmx"))
    for lmx_path in tqdm(lmx_files, desc="Converting predicted LMX -> XML"):
        try:
            lmx_text = lmx_path.read_text(encoding="utf-8").strip()
            xml_text = tokenizer.converter.delinearize(lmx_text)
            out_path = pred_xml_dir / f"{lmx_path.stem}.xml"
            out_path.write_text(xml_text, encoding="utf-8")
            converted += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": lmx_path.name, "error": str(exc)})

    return {
        "num_lmx_files": len(lmx_files),
        "num_xml_converted": converted,
        "num_xml_failed": len(failures),
        "failures": failures,
    }


def run_command(command: list[str], *, cwd: Path) -> None:
    print("[CMD]", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


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


def main() -> None:
    args = build_parser().parse_args()

    eval_hf_dataset = args.eval_root / "hf_dataset"
    gt_xml_dir = args.eval_root / "musicxml"

    if not eval_hf_dataset.exists():
        raise FileNotFoundError(f"Missing evaluation HF dataset: {eval_hf_dataset}")
    if not gt_xml_dir.exists():
        raise FileNotFoundError(f"Missing truncated MusicXML directory: {gt_xml_dir}")

    work_dir = Path(args.work_dir)
    pred_lmx_dir = work_dir / "pred_lmx"
    pred_xml_dir = work_dir / "pred_xml"
    report_json = args.report_json or (work_dir / "our_model_eval_report.json")
    conversion_log_path = work_dir / "lmx_to_xml_conversion_log.json"
    precheck_path = work_dir / "alignment_precheck.json"

    generated_config_path = save_temp_config_for_eval(
        base_config_path=args.base_config,
        eval_hf_dataset_path=eval_hf_dataset,
        output_path=work_dir / "config_for_eval.yaml",
        tokenizer_path=args.tokenizer_path,
    )

    if not args.skip_pred:
        command = [
            args.python_bin,
            "midi2score/pred_seq2seq.py",
            "--config",
            str(generated_config_path),
            "--ckpt",
            str(args.ckpt),
            "--out",
            str(pred_lmx_dir),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
        ]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        run_command(command, cwd=PROJECT_ROOT)

    cfg = yaml.safe_load(generated_config_path.read_text(encoding="utf-8"))
    tokenizer_path = args.tokenizer_path or Path(cfg["data"]["tokenizer_path"])

    tokenizer = MusicXMLTokenizer()
    tokenizer.load_bpe_model(str(tokenizer_path))
    tokenizer.eval_mode()

    conversion_log = convert_lmx_dir_to_xml(pred_lmx_dir=pred_lmx_dir, pred_xml_dir=pred_xml_dir, tokenizer=tokenizer)
    conversion_log_path.parent.mkdir(parents=True, exist_ok=True)
    conversion_log_path.write_text(json.dumps(conversion_log, ensure_ascii=False, indent=2), encoding="utf-8")

    alignment = check_alignment(gt_xml_dir=gt_xml_dir, pred_xml_dir=pred_xml_dir)
    precheck_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_eval:
        eval_command = [
            args.python_bin,
            "evaluation.py",
            "--pred_xml_dir",
            str(pred_xml_dir),
            "--gt_xml_dir",
            str(gt_xml_dir),
            "--save_json",
            str(report_json),
        ]
        run_command(eval_command, cwd=PROJECT_ROOT)

    print("=== Done ===")
    print(f"Working directory     : {work_dir}")
    print(f"Generated config      : {generated_config_path}")
    print(f"Predicted LMX dir     : {pred_lmx_dir}")
    print(f"Predicted XML dir     : {pred_xml_dir}")
    print(f"Alignment precheck    : {precheck_path}")
    print(f"Conversion log        : {conversion_log_path}")
    print(f"Evaluation report     : {report_json}")


if __name__ == "__main__":
    main()
