from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Allow `python midi2score/pred_seq2seq.py ...` to resolve project imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midi2score.model_seq2seq import TransformerForConditionalGeneration
from midi2score.config import load_seq2seq_config
from evaluation import (
    aggregate_xml_results,
    evaluate_xml_pair,
)

from tokenizer.cpword_tokenizer_config import cpword_tokenizer
from tokenizer.musicxml_tokenizer import MusicXMLTokenizer

KNOWN_VARIANTS = ("clean", "light", "heavy")

REQUIRED_EVAL_COLUMNS = {
    "id",
    "selected_variant",
    "selected_cpword_ids",
    "selected_source_length",
    "truncated_lmx_path",
    "truncated_musicxml_path",
}


@dataclass(slots=True)
class EvalSample:
    sample_id: str
    variant: str
    cpword_ids: list[list[int]]
    source_length: int
    gt_lmx_path: Path
    gt_xml_path: Path


@dataclass(slots=True)
class PredictionRecord:
    sample_id: str
    variant: str
    pred_lmx_path: Path | None
    pred_xml_path: Path | None
    gt_lmx_path: Path | None
    gt_xml_path: Path
    lmx_error: str | None
    xml_error: str | None


@dataclass(slots=True)
class InferenceArtifacts:
    records: list[PredictionRecord]
    variant_counter: dict[str, int]
    peak_memory_mb: float | None


# =========================
# 1. helpers
# =========================
def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def sanitize_sample_id(sample_id: str) -> str:
    sanitized = sample_id.replace("/", "_").replace("\\", "_").strip()
    return sanitized or "sample"


def load_tokenizer(tokenizer_path: str | Path) -> MusicXMLTokenizer:
    tokenizer = MusicXMLTokenizer()
    tokenizer.load_bpe_model(str(tokenizer_path))
    tokenizer.eval_mode()
    return tokenizer


def trim_token_ids(
    token_ids: list[int],
    *,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> list[int]:
    cleaned: list[int] = []

    for token_id in token_ids:
        # skip leading BOS
        if token_id == bos_token_id and not cleaned:
            continue

        # stop at EOS or PAD
        if token_id in {eos_token_id, pad_token_id}:
            break

        cleaned.append(token_id)

    return cleaned


def decode_tokens_to_lmx(token_ids: list[int], tokenizer: MusicXMLTokenizer) -> str:
    if not token_ids:
        return ""

    # Important:
    # do NOT use generic HuggingFace tokenizer.decode here.
    # This project needs MusicXMLTokenizer -> BPE decode -> mapper.decode_to_lmx.
    byte_str = tokenizer.bpe.decode(token_ids)
    text = tokenizer.mapper.decode_to_lmx(byte_str)
    return text.strip()


def resolve_cpword_bos_eos_tokens() -> tuple[list[int], list[int]]:
    vocab = cpword_tokenizer.vocab
    if not isinstance(vocab, list) or not vocab:
        raise RuntimeError("CPWord tokenizer vocab is not in expected multi-vocabulary format.")

    bos_token: list[int] = []
    eos_token: list[int] = []
    for dim_vocab in vocab:
        if not isinstance(dim_vocab, dict):
            raise RuntimeError("CPWord tokenizer dimension vocab is not a dictionary.")

        bos_id = dim_vocab.get("BOS_None")
        eos_id = dim_vocab.get("EOS_None")
        if bos_id is None or eos_id is None:
            raise RuntimeError("Failed to resolve CPWord BOS_None/EOS_None IDs.")

        bos_token.append(int(bos_id))
        eos_token.append(int(eos_id))

    return bos_token, eos_token


def strip_cpword_special_tokens(
    token_ids: list[list[int]],
    *,
    bos_token: list[int],
    eos_token: list[int],
) -> list[list[int]]:
    if not token_ids:
        return token_ids

    start = 1 if list(token_ids[0]) == bos_token else 0
    end = len(token_ids) - 1 if len(token_ids) > start and list(token_ids[-1]) == eos_token else len(token_ids)
    return token_ids[start:end]


# =========================
# 2. I/O helpers
# =========================
def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_eval_split(hf_dataset_path: Path, split: str) -> Dataset:
    dataset_dict = load_from_disk(str(hf_dataset_path))
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError(f"Expected DatasetDict at {hf_dataset_path}, got {type(dataset_dict)}")

    if split not in dataset_dict:
        raise ValueError(f"Split {split!r} not found. Available splits: {list(dataset_dict.keys())}")

    split_dataset = dataset_dict[split]
    missing = sorted(REQUIRED_EVAL_COLUMNS.difference(split_dataset.column_names))
    if missing:
        raise ValueError(
            f"Evaluation dataset split {split!r} missing required columns: {missing}"
        )

    return split_dataset


def build_eval_samples(
    split_dataset: Dataset,
    *,
    eval_root: Path,
    max_samples: int | None,
) -> list[EvalSample]:
    samples: list[EvalSample] = []

    limit = len(split_dataset)
    if max_samples is not None:
        limit = min(limit, max_samples)

    for idx in range(limit):
        row = split_dataset[idx]

        sample_id = str(row["id"])
        if not sample_id:
            raise ValueError(f"Row {idx} has empty id")

        variant = str(row["selected_variant"])
        if variant not in KNOWN_VARIANTS:
            raise ValueError(
                f"Row {idx} has unsupported selected_variant={variant!r}. "
                f"Expected one of {KNOWN_VARIANTS}."
            )

        cpword_ids = row["selected_cpword_ids"]
        if not cpword_ids:
            raise ValueError(f"Row {idx} (id={sample_id}) has empty selected_cpword_ids")

        source_length = int(row["selected_source_length"])
        if source_length <= 0:
            raise ValueError(f"Row {idx} (id={sample_id}) has non-positive selected_source_length={source_length}")

        xml_rel = str(row["truncated_musicxml_path"])
        if not xml_rel:
            raise ValueError(f"Row {idx} (id={sample_id}) has empty truncated_musicxml_path")
        gt_xml_path = eval_root / xml_rel
        if not gt_xml_path.exists():
            raise FileNotFoundError(
                f"Row {idx} (id={sample_id}) references missing GT xml path: {gt_xml_path}"
            )

        lmx_rel = str(row["truncated_lmx_path"])
        if not lmx_rel:
            raise ValueError(f"Row {idx} (id={sample_id}) has empty truncated_lmx_path")
        gt_lmx_path = eval_root / lmx_rel
        if not gt_lmx_path.exists():
            raise FileNotFoundError(
                f"Row {idx} (id={sample_id}) references missing GT lmx path: {gt_lmx_path}"
            )

        samples.append(
            EvalSample(
                sample_id=sample_id,
                variant=variant,
                cpword_ids=cpword_ids,
                source_length=source_length,
                gt_lmx_path=gt_lmx_path,
                gt_xml_path=gt_xml_path,
            )
        )

    return samples


def build_inference_batches(
    samples: list[EvalSample],
    *,
    batch_size: int,
    length_bucketing: bool,
) -> list[list[int]]:
    indices = list(range(len(samples)))
    if length_bucketing:
        indices.sort(key=lambda i: samples[i].source_length, reverse=True)

    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def make_encoder_batch(
    batch_samples: list[EvalSample],
    *,
    pad_token_id: int,
    cpword_bos_token: list[int],
    cpword_eos_token: list[int],
    max_source_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors: list[torch.Tensor] = []
    feature_dim: int | None = None
    max_source_content_length = max_source_length - 2

    if max_source_content_length <= 0:
        raise ValueError("max_source_length must be >= 2 to include CPWord BOS/EOS")

    for sample in batch_samples:
        stripped = strip_cpword_special_tokens(
            sample.cpword_ids,
            bos_token=cpword_bos_token,
            eos_token=cpword_eos_token,
        )
        trimmed_content = stripped[:max_source_content_length]
        trimmed = [cpword_bos_token, *trimmed_content, cpword_eos_token]
        if not trimmed:
            raise ValueError(f"Sample {sample.sample_id} has empty selected_cpword_ids")

        tensor = torch.tensor(trimmed, dtype=torch.long)
        if tensor.dim() != 2:
            raise ValueError(
                f"Sample {sample.sample_id} has invalid CPWord shape: {tuple(tensor.shape)}"
            )

        if feature_dim is None:
            feature_dim = tensor.shape[1]
        elif tensor.shape[1] != feature_dim:
            raise ValueError(
                f"Sample {sample.sample_id} feature dim mismatch: {tensor.shape[1]} != {feature_dim}"
            )

        tensors.append(tensor)

    assert feature_dim is not None

    max_len = max(t.size(0) for t in tensors)
    batch_size = len(tensors)

    encoder_tokens = torch.full(
        (batch_size, max_len, feature_dim),
        pad_token_id,
        dtype=torch.long,
    )
    encoder_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, tensor in enumerate(tensors):
        length = tensor.size(0)
        encoder_tokens[i, :length, :] = tensor
        encoder_padding_mask[i, :length] = False

    return encoder_tokens.to(device), encoder_padding_mask.to(device)


def resolve_autocast_dtype(dtype_name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None

    if dtype_name == "fp32":
        return None
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    raise ValueError(f"Unsupported dtype mode: {dtype_name}")


def get_amp_context(autocast_dtype: torch.dtype | None):
    if autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


# =========================
# 3. load checkpoint
# =========================
def iter_key_candidates(raw_key: str) -> list[str]:
    prefixes = ("state_dict.", "_orig_mod.", "module.", "model.")

    queue = [raw_key]
    ordered: list[str] = []
    seen: set[str] = set()

    while queue:
        key = queue.pop(0)
        if key in seen:
            continue

        seen.add(key)
        ordered.append(key)

        for prefix in prefixes:
            if key.startswith(prefix):
                queue.append(key[len(prefix):])

        if key.startswith("model."):
            queue.append(key[len("model."):])
        else:
            queue.append(f"model.{key}")

    return ordered


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config,
    device: torch.device,
) -> TransformerForConditionalGeneration:
    model = TransformerForConditionalGeneration(config.model)

    training_mode = getattr(config.training, "training_mode", None)
    use_lora = training_mode == "lora" or getattr(config.training, "use_lora", False)

    if use_lora:
        peft_config = LoraConfig(
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "feedforward.linear1",
                "feedforward.linear2",
            ],
            lora_dropout=config.training.lora_dropout,
            bias="none",
        )
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
    else:
        raw_state_dict = ckpt

    if not isinstance(raw_state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a valid state dict. "
            f"Got type: {type(raw_state_dict)}"
        )

    expected_state_dict = model.state_dict()
    cleaned_state_dict: dict[str, torch.Tensor] = {}
    unmatched_keys: list[str] = []

    for raw_key, value in raw_state_dict.items():
        matched = False
        for candidate_key in iter_key_candidates(raw_key):
            if candidate_key in expected_state_dict:
                cleaned_state_dict[candidate_key] = value
                matched = True
                break

        if not matched:
            unmatched_keys.append(raw_key)

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    if unmatched_keys:
        print(f"[WARN] Unmatched raw checkpoint keys: {len(unmatched_keys)}")

    model.to(device)
    model.eval()
    return model


# =========================
# 4. per-sample conversion
# =========================
def convert_pred_to_lmx_and_xml(
    pred_ids: list[int],
    *,
    tokenizer: MusicXMLTokenizer,
    sample_id: str,
    lmx_dir: Path,
    xml_dir: Path,
) -> tuple[Path | None, Path | None, str | None, str | None]:
    safe_id = sanitize_sample_id(sample_id)

    pred_lmx_path = lmx_dir / f"{safe_id}.lmx"
    pred_xml_path = xml_dir / f"{safe_id}.xml"

    lmx_error: str | None = None
    xml_error: str | None = None

    lmx_text = ""
    try:
        lmx_text = decode_tokens_to_lmx(pred_ids, tokenizer)
        save_text(lmx_text, pred_lmx_path)
    except Exception as exc:  # noqa: BLE001
        lmx_error = str(exc)
        pred_lmx_path = None

    if lmx_error is None:
        try:
            xml_text = tokenizer.converter.delinearize(lmx_text)
            save_text(xml_text, pred_xml_path)
        except Exception as exc:  # noqa: BLE001
            xml_error = str(exc)
            pred_xml_path = None
    else:
        pred_xml_path = None

    return pred_lmx_path, pred_xml_path, lmx_error, xml_error


# =========================
# 5. inference
# =========================
def run_inference_on_eval_dataset(
    *,
    config_path: Path,
    checkpoint_path: Path,
    eval_root: Path,
    hf_dataset_path: Path,
    split: str,
    output_root: Path,
    batch_size: int,
    max_samples: int | None,
    max_source_length: int,
    max_target_length: int,
    temperature: float = 1.0,
    top_k: int | None = 1,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    dtype_mode: str = "auto",
    length_bucketing: bool = True,
) -> InferenceArtifacts:
    config = load_seq2seq_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )

    tokenizer_path = config.data.tokenizer_path
    if tokenizer_path is None:
        raise ValueError("config.data.tokenizer_path is required for decoding BPE LMX")

    tokenizer = load_tokenizer(resolve_path(tokenizer_path))

    split_dataset = load_eval_split(hf_dataset_path, split=split)
    samples = build_eval_samples(
        split_dataset,
        eval_root=eval_root,
        max_samples=max_samples,
    )
    if not samples:
        raise ValueError("No samples found in evaluation dataset")

    lmx_dir = output_root / "lmx"
    xml_dir = output_root / "musicxml"
    lmx_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    bos_token_id = config.model.decoder_config.bos_token_id
    eos_token_id = config.model.decoder_config.eos_token_id
    pad_token_id = config.model.decoder_config.pad_token_id

    if top_k is not None and top_k <= 0:
        top_k = None

    autocast_dtype = resolve_autocast_dtype(dtype_mode, device)
    cpword_bos_token, cpword_eos_token = resolve_cpword_bos_eos_tokens()
    batches = build_inference_batches(
        samples,
        batch_size=batch_size,
        length_bucketing=length_bucketing,
    )

    variant_counter = Counter(sample.variant for sample in samples)
    records: list[PredictionRecord] = []

    print("\n===== Running Inference =====\n")
    print(f"[INFO] tokenizer_path = {resolve_path(tokenizer_path)}")
    print(f"[INFO] eval_root       = {eval_root.resolve()}")
    print(f"[INFO] hf_dataset      = {hf_dataset_path.resolve()}")
    print(f"[INFO] split           = {split}")
    print(f"[INFO] num_samples     = {len(samples)}")
    print(f"[INFO] output_dir      = {output_root.resolve()}")
    print(f"[INFO] device          = {device}")
    print(f"[INFO] dtype_mode      = {dtype_mode} (autocast={autocast_dtype})")
    print(f"[INFO] batch_size      = {batch_size}")
    print(f"[INFO] length_bucketing= {length_bucketing}")
    print(f"[INFO] max_source_len  = {max_source_length}")
    print(f"[INFO] max_target_len  = {max_target_length}")
    print(f"[INFO] temperature     = {temperature}")
    print(f"[INFO] top_k           = {top_k}")
    print(f"[INFO] top_p           = {top_p}")
    print(f"[INFO] repetition_penalty = {repetition_penalty}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(batches, desc="Generating")

    with torch.inference_mode():
        for batch_indices in progress:
            batch_samples = [samples[i] for i in batch_indices]

            try:
                encoder_input_tokens, encoder_padding_mask = make_encoder_batch(
                    batch_samples,
                    pad_token_id=pad_token_id,
                    cpword_bos_token=cpword_bos_token,
                    cpword_eos_token=cpword_eos_token,
                    max_source_length=max_source_length,
                    device=device,
                )

                with get_amp_context(autocast_dtype):
                    preds = model.generate(
                        encoder_input_tokens=encoder_input_tokens,
                        encoder_padding_mask=encoder_padding_mask,
                        max_length=max_target_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
            except Exception as exc:  # noqa: BLE001
                generation_error = f"Generation failed: {exc}"
                for sample in batch_samples:
                    records.append(
                        PredictionRecord(
                            sample_id=sample.sample_id,
                            variant=sample.variant,
                            pred_lmx_path=None,
                            pred_xml_path=None,
                            gt_lmx_path=sample.gt_lmx_path,
                            gt_xml_path=sample.gt_xml_path,
                            lmx_error=generation_error,
                            xml_error=generation_error,
                        )
                    )
                continue

            for i, sample in enumerate(batch_samples):
                try:
                    pred_ids = trim_token_ids(
                        preds[i].tolist(),
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    records.append(
                        PredictionRecord(
                            sample_id=sample.sample_id,
                            variant=sample.variant,
                            pred_lmx_path=None,
                            pred_xml_path=None,
                            gt_lmx_path=sample.gt_lmx_path,
                            gt_xml_path=sample.gt_xml_path,
                            lmx_error=f"Failed to parse generated tokens: {exc}",
                            xml_error=f"Failed to parse generated tokens: {exc}",
                        )
                    )
                    continue

                pred_lmx_path, pred_xml_path, lmx_error, xml_error = convert_pred_to_lmx_and_xml(
                    pred_ids,
                    tokenizer=tokenizer,
                    sample_id=sample.sample_id,
                    lmx_dir=lmx_dir,
                    xml_dir=xml_dir,
                )

                records.append(
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        variant=sample.variant,
                        pred_lmx_path=pred_lmx_path,
                        pred_xml_path=pred_xml_path,
                        gt_lmx_path=sample.gt_lmx_path,
                        gt_xml_path=sample.gt_xml_path,
                        lmx_error=lmx_error,
                        xml_error=xml_error,
                    )
                )


    peak_memory_mb: float | None = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"\nSaved predicted LMX to: {lmx_dir.resolve()}")
    print(f"Saved predicted XML to: {xml_dir.resolve()}")

    if peak_memory_mb is not None:
        print(f"Peak CUDA memory allocated: {peak_memory_mb:.2f} MB")

    return InferenceArtifacts(
        records=records,
        variant_counter=dict(variant_counter),
        peak_memory_mb=peak_memory_mb,
    )


# =========================
# 6. evaluation
# =========================
def _append_to_group(
    grouped: dict[str, list[dict[str, Any]]],
    variant: str,
    result: dict[str, Any],
) -> None:
    grouped["overall"].append(result)
    if variant in KNOWN_VARIANTS:
        grouped[variant].append(result)


def evaluate_records(records: list[PredictionRecord],*,onset_tol: float = 0.0,duration_tol: float = 0.0,) -> dict[str, Any]:
    group_names = ["overall", *KNOWN_VARIANTS]

    grouped_xml_results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    xml_failures: list[dict[str, str]] = []

    xml_failures_by_group: dict[str, list[dict[str, str]]] = {k: [] for k in group_names}

    xml_available_counter: Counter[str] = Counter()

    def record_groups(variant: str) -> list[str]:
        if variant in KNOWN_VARIANTS:
            return ["overall", variant]
        return ["overall"]

    for record in tqdm(records, desc="Evaluating records"):
        groups = record_groups(record.variant)


        has_xml_pair = (
            record.pred_xml_path is not None
            and record.gt_xml_path is not None
            and record.pred_xml_path.exists()
            and record.gt_xml_path.exists()
        )
        if has_xml_pair:
            for group in groups:
                xml_available_counter[group] += 1
            try:
                result = evaluate_xml_pair(str(record.pred_xml_path),str(record.gt_xml_path),onset_tol=onset_tol,duration_tol=duration_tol,)
                result["id"] = record.sample_id
                result["variant"] = record.variant
                _append_to_group(grouped_xml_results, record.variant, result)
            except Exception as exc:  # noqa: BLE001
                failure = {"id": record.sample_id, "error": str(exc)}
                xml_failures.append(failure)
                for group in groups:
                    xml_failures_by_group[group].append(failure)
    xml_payload: dict[str, Any] = {}

    for group in group_names:

        xml_results = grouped_xml_results.get(group, [])


        xml_payload[group] = {
            "summary": aggregate_xml_results(xml_results),
            "num_available_pairs": int(xml_available_counter.get(group, 0)),
            "num_evaluated_pairs": len(xml_results),
            "failures": xml_failures_by_group[group],
        }

    return {
        "xml": xml_payload,
        "global_failures": {
            "xml": xml_failures,
        },
    }


def format_metrics_block(title: str, metrics: dict[str, Any]) -> list[str]:
    lines = [title]
    for group in ["overall", *KNOWN_VARIANTS]:
        group_payload = metrics.get(group, {})
        summary = group_payload.get("summary", {})
        n_available = group_payload.get("num_available_pairs", 0)
        n_eval = group_payload.get("num_evaluated_pairs", 0)

        lines.append(f"[{group}] available={n_available} evaluated={n_eval}")
        if not summary:
            lines.append("  summary: <empty>")
            continue

        for key, value in summary.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")

    return lines


def write_evaluation_logs(
    *,
    output_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    eval_root: Path,
    hf_dataset_path: Path,
    split: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    temperature: float,
    top_k: int | None,
    dtype_mode: str,
    length_bucketing: bool,
    onset_tol: float,
    duration_tol: float,
    inference_artifacts: InferenceArtifacts,
    evaluation_payload: dict[str, Any],
) -> None:
    txt_path = output_root / "evaluation.txt"
    json_path = output_root / "evaluation.json"

    generation_fail_xml = sum(1 for r in inference_artifacts.records if r.xml_error is not None)

    lines: list[str] = []
    lines.append("================ Evaluation Pipeline Report ================")
    lines.append("")
    lines.append("[Run Configuration]")
    lines.append(f"config: {config_path.resolve()}")
    lines.append(f"checkpoint: {checkpoint_path.resolve()}")
    lines.append(f"eval_root: {eval_root.resolve()}")
    lines.append(f"hf_dataset_path: {hf_dataset_path.resolve()}")
    lines.append(f"split: {split}")
    lines.append(f"batch_size: {batch_size}")
    lines.append(f"max_source_length: {max_source_length}")
    lines.append(f"max_target_length: {max_target_length}")
    lines.append(f"temperature: {temperature}")
    lines.append(f"top_k: {top_k}")
    lines.append(f"dtype_mode: {dtype_mode}")
    lines.append(f"length_bucketing: {length_bucketing}")
    lines.append(f"onset_tol: {onset_tol}")
    lines.append(f"duration_tol: {duration_tol}")
    lines.append("")

    lines.append("[Generation Summary]")
    lines.append(f"num_samples: {len(inference_artifacts.records)}")
    lines.append(f"variant_counts: {inference_artifacts.variant_counter}")
    lines.append(f"xml_generation_failures: {generation_fail_xml}")
    if inference_artifacts.peak_memory_mb is not None:
        lines.append(f"peak_cuda_memory_mb: {inference_artifacts.peak_memory_mb:.2f}")
    lines.append("")

    lines.extend(format_metrics_block("[XML Metrics]", evaluation_payload["xml"]))
    lines.append("")
    lines.append("[Global Evaluation Failures]")
    lines.append(f"xml_failures: {len(evaluation_payload['global_failures']['xml'])}")

    report_text = "\n".join(lines) + "\n"

    save_text(report_text, txt_path)

    payload = {
        "config": {
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "eval_root": str(eval_root),
            "hf_dataset_path": str(hf_dataset_path),
            "split": split,
            "batch_size": batch_size,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "temperature": temperature,
            "top_k": top_k,
            "dtype_mode": dtype_mode,
            "length_bucketing": length_bucketing,
            "onset_tol": onset_tol,
            "duration_tol": duration_tol,
        },
        "generation": {
            "num_samples": len(inference_artifacts.records),
            "variant_counts": inference_artifacts.variant_counter,
            "peak_cuda_memory_mb": inference_artifacts.peak_memory_mb,
            "errors": {
                "xml_generation_failures": generation_fail_xml,
            },
        },
        "evaluation": evaluation_payload,
    }

    save_text(json.dumps(payload, ensure_ascii=False, indent=2), json_path)

    print("\n" + report_text)
    print(f"Saved evaluation text report to: {txt_path.resolve()}")
    print(f"Saved evaluation json report to: {json_path.resolve()}")


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_seq2seq_config(config_path)

    if args.ckpt:
        checkpoint_path = resolve_path(args.ckpt)
    else:
        default_ckpt = getattr(config.training, "save_best_checkpoint_path", None)
        if not default_ckpt:
            raise ValueError(
                "--ckpt not provided and config.training.save_best_checkpoint_path is missing"
            )
        checkpoint_path = resolve_path(default_ckpt)

    if args.out_dir:
        output_root = resolve_path(args.out_dir)
    else:
        output_root = checkpoint_path.parent / "eval_result"

    output_root.mkdir(parents=True, exist_ok=True)

    eval_root = resolve_path(args.eval_root)
    hf_dataset_path = resolve_path(args.hf_dataset_path) if args.hf_dataset_path else (eval_root / "hf_dataset")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root not found: {eval_root}")
    if not hf_dataset_path.exists():
        raise FileNotFoundError(f"HF dataset path not found: {hf_dataset_path}")

    max_source_length = args.max_source_length or config.data.max_source_length
    max_target_length = args.max_target_length or config.data.max_target_length

    inference_artifacts = run_inference_on_eval_dataset(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        eval_root=eval_root,
        hf_dataset_path=hf_dataset_path,
        split=args.split,
        output_root=output_root,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        temperature=args.temperature,
        top_k=args.top_k,
        dtype_mode=args.dtype,
        length_bucketing=not args.disable_length_bucketing,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    evaluation_payload = evaluate_records(inference_artifacts.records, onset_tol=args.onset_tol, duration_tol=args.duration_tol)

    write_evaluation_logs(
        output_root=output_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        eval_root=eval_root,
        hf_dataset_path=hf_dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        temperature=args.temperature,
        top_k=args.top_k,
        dtype_mode=args.dtype,
        length_bucketing=not args.disable_length_bucketing,
        inference_artifacts=inference_artifacts,
        evaluation_payload=evaluation_payload,
        onset_tol=args.onset_tol,
        duration_tol=args.duration_tol,
        
    )

# =========================
# 7. CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run seq2seq prediction on eval dataset (selected_cpword_ids), export LMX/XML outputs, "
            "and compute grouped clean/light/heavy/overall evaluation metrics."
        )
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to config.training.save_best_checkpoint_path",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <checkpoint_parent>/eval_result",
    )

    parser.add_argument("--eval-root", type=str, default="DATA/eval_dataset")
    parser.add_argument(
        "--hf-dataset-path",
        type=str,
        default=None,
        help="Optional direct path to evaluation HF DatasetDict. Defaults to <eval-root>/hf_dataset",
    )
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--onset-tol", type=float, default=0.0)
    parser.add_argument("--duration-tol", type=float, default=0.0)

    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--disable-length-bucketing", action="store_true")

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_source_length is not None and args.max_source_length <= 0:
        raise ValueError("--max-source-length must be positive")
    if args.max_target_length is not None and args.max_target_length <= 0:
        raise ValueError("--max-target-length must be positive")
    if args.top_p is not None and not 0 <= args.top_p <= 1:
        raise ValueError("--top-p must be a float between 0 and 1")
    if args.repetition_penalty is not None and args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be a positive float")

    run_pipeline(args)