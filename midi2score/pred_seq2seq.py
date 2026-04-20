from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Allow `python midi2score/pred_seq2seq.py ...` to resolve project imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midi2score.model_seq2seq import TransformerForConditionalGeneration
from midi2score.data_seq2seq import build_seq2seq_dataloader
from midi2score.config import load_seq2seq_config

try:
    from tokenizer.musicxml_tokenizer import MusicXMLTokenizer
except ModuleNotFoundError as exc:
    if exc.name == "lmx":
        raise RuntimeError(
            "Failed to import MusicXMLTokenizer because dependency 'lmx' is missing.\n"
            "This pred script must run in the SAME project environment used by the tokenizer.\n"
            "Please run with your project venv/interpreter, e.g.:\n"
            "  ./.venv/bin/python midi2score/pred_seq2seq.py --config ... --ckpt ... --out ...\n"
            "or install the tokenizer dependency set into the current environment."
        ) from exc
    raise


# =========================
# 1. tokenizer helper
# =========================
def load_tokenizer(tokenizer_path: str) -> MusicXMLTokenizer:
    tokenizer = MusicXMLTokenizer()
    tokenizer.load_bpe_model(tokenizer_path)
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


# =========================
# 2. save file
# =========================
def save_lmx(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# =========================
# 3. load checkpoint
# =========================
def load_model_from_checkpoint(
    checkpoint_path: str,
    config,
    device: torch.device,
) -> TransformerForConditionalGeneration:
    model = TransformerForConditionalGeneration(config.model)

    if getattr(config.training, "use_lora", False):
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

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
    else:
        raw_state_dict = ckpt

    expected_state_dict = model.state_dict()
    cleaned_state_dict: dict[str, torch.Tensor] = {}

    for raw_key, value in raw_state_dict.items():
        candidate_keys = [raw_key]

        if raw_key.startswith("_orig_mod."):
            candidate_keys.append(raw_key[len("_orig_mod."):])

        if raw_key.startswith("model."):
            candidate_keys.append(raw_key[len("model."):])
        else:
            candidate_keys.append(f"model.{raw_key}")

        for candidate_key in candidate_keys:
            if candidate_key in expected_state_dict:
                cleaned_state_dict[candidate_key] = value
                break

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


# =========================
# 4. optional sanity print
# =========================
def maybe_print_sanity_example(
    *,
    batch_idx: int,
    sample_idx: int,
    pred_ids: list[int],
    batch,
    tokenizer: MusicXMLTokenizer,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    if batch_idx != 0 or sample_idx != 0:
        return

    print("\n===== Sanity Check: first sample =====")

    try:
        pred_text = decode_tokens_to_lmx(pred_ids, tokenizer)
        print("[pred first 300 chars]")
        print(pred_text[:300] if pred_text else "<EMPTY>")
    except Exception as exc:
        print(f"[WARN] Failed to decode prediction: {exc}")

    if hasattr(batch, "labels"):
        label_ids = batch.labels[sample_idx].tolist()
        label_ids = [tid for tid in label_ids if tid != -100]
        label_ids = trim_token_ids(
            label_ids,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        try:
            gt_text = decode_tokens_to_lmx(label_ids, tokenizer)
            print("[gt first 300 chars]")
            print(gt_text[:300] if gt_text else "<EMPTY>")
        except Exception as exc:
            print(f"[WARN] Failed to decode ground truth: {exc}")

    print("======================================\n")


# =========================
# 5. main inference
# =========================
def run_inference(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    max_samples: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = 1,
) -> None:
    config = load_seq2seq_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )

    tokenizer = load_tokenizer(config.data.tokenizer_path)

    test_data_config = config.data
    test_data_config.split = "test"

    test_loader = build_seq2seq_dataloader(
        test_data_config,
        batch_size=1,
        shuffle=False,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    bos_token_id = config.model.decoder_config.bos_token_id
    eos_token_id = config.model.decoder_config.eos_token_id
    pad_token_id = config.model.decoder_config.pad_token_id

    print("\n===== Running Inference =====\n")
    print(f"[INFO] tokenizer_path = {config.data.tokenizer_path}")
    print(f"[INFO] output_dir      = {output_root.resolve()}")
    print(f"[INFO] device          = {device}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if max_samples is not None and batch_idx >= max_samples:
                break

            batch = batch.to(device)

            # KV cache optimization lives inside model.generate()
            
            preds = model.generate(
                encoder_input_tokens=batch.encoder_input_tokens,
                encoder_padding_mask=batch.encoder_padding_mask,
                max_length=config.data.max_target_length,
                temperature=temperature,
                top_k=top_k,
            )

            for i in range(preds.size(0)):
                pred_ids = trim_token_ids(
                    preds[i].tolist(),
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

                maybe_print_sanity_example(
                    batch_idx=batch_idx,
                    sample_idx=i,
                    pred_ids=pred_ids,
                    batch=batch,
                    tokenizer=tokenizer,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

                lmx_text = decode_tokens_to_lmx(pred_ids, tokenizer)

                if hasattr(batch, "file_names"):
                    filename = batch.file_names[i]
                else:
                    filename = f"sample_{batch_idx}_{i}"

                save_path = output_root / f"{filename}.lmx"
                save_lmx(lmx_text, save_path)

    print(f"\n✅ Saved predictions to: {output_root.resolve()}")


# =========================
# 6. CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)

    args = parser.parse_args()

    run_inference(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        output_dir=args.out,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_k=args.top_k,
    )