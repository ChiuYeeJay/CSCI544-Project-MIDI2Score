"""Transcribe one MIDI file to MusicXML with a trained MIDI2Score checkpoint.

Example:
    python transcribe.py \
      --midi input.mid \
      --checkpoint checkpoints/best.pt \
      --tokenizer tokenizer/tokenizer.json \
      --config configs/experiments/seq2seq_fullft_32.yaml \
      --output output.musicxml

The checkpoint stores weights only. ``--config`` must be the YAML that matches
that checkpoint's architecture. ``--tokenizer`` is the LMX BPE tokenizer
(tokenizer.json). MIDI is encoded with the CPWord tokenizer defined in this
repository.

Sequences longer than ``max_source_length`` are cut from the front. The written
score then covers only that prefix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {name}")


def encode_midi(midi_path: Path, cpword_tokenizer) -> list[list[int]]:
    import symusic

    encoded = cpword_tokenizer.encode(symusic.Score(str(midi_path)))
    if hasattr(encoded, "ids"):
        ids = encoded.ids
    elif isinstance(encoded, list) and encoded and hasattr(encoded[0], "ids"):
        raise ValueError(
            f"CPWord encoder returned {len(encoded)} token streams. "
            "This tool expects a single stream."
        )
    else:
        ids = encoded

    rows = [[int(value) for value in row] for row in ids]
    if not rows:
        raise ValueError(f"MIDI file produced an empty CPWord sequence: {midi_path}")
    return rows


def cpword_ids_to_batch(
    token_ids: list[list[int]],
    *,
    cpword_bos_token: list[int],
    cpword_eos_token: list[int],
    max_source_length: int,
    device: torch.device,
    strip_cpword_special_tokens,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids and len(token_ids[0]) != len(cpword_bos_token):
        raise ValueError(
            f"Unexpected CPWord token width {len(token_ids[0])}, "
            f"expected {len(cpword_bos_token)}"
        )

    stripped = strip_cpword_special_tokens(
        token_ids,
        bos_token=cpword_bos_token,
        eos_token=cpword_eos_token,
    )
    if not stripped:
        raise ValueError("MIDI file produced an empty CPWord sequence.")

    max_content_length = max_source_length - 2
    if max_content_length <= 0:
        raise ValueError("max_source_length must be >= 2 to include CPWord BOS/EOS")

    if len(stripped) > max_content_length:
        print(
            f"[WARN] MIDI has {len(stripped)} CPWord tokens; "
            f"keeping the first {max_content_length}. The score will be incomplete."
        )
        stripped = stripped[:max_content_length]

    trimmed = [cpword_bos_token, *stripped, cpword_eos_token]
    tokens = torch.tensor(trimmed, dtype=torch.long, device=device).unsqueeze(0)
    padding_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=device)
    return tokens, padding_mask


def transcribe(args: argparse.Namespace) -> None:
    midi_path = args.midi.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else midi_path.with_suffix(".musicxml")
    )

    for label, path in (
        ("MIDI", midi_path),
        ("checkpoint", checkpoint_path),
        ("tokenizer", tokenizer_path),
        ("config", config_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    device = select_device(args.device)

    # Importing midi2score builds the CPWord tokenizer and emits a miditok warning.
    # Keep that off the --help path and off argument/path failures.
    from midi2score.config import load_seq2seq_config
    from midi2score.pred_seq2seq import (
        decode_tokens_to_lmx,
        get_amp_context,
        load_model_from_checkpoint,
        load_tokenizer,
        resolve_autocast_dtype,
        resolve_cpword_bos_eos_tokens,
        strip_cpword_special_tokens,
        trim_token_ids,
    )
    from tokenizer.cpword_tokenizer_config import cpword_tokenizer

    config = load_seq2seq_config(config_path)
    max_source_length = args.max_source_length or config.data.max_source_length
    max_target_length = args.max_target_length or config.data.max_target_length

    model = load_model_from_checkpoint(checkpoint_path, config, device)
    tokenizer = load_tokenizer(tokenizer_path)

    cpword_ids = encode_midi(midi_path, cpword_tokenizer)
    cpword_bos_token, cpword_eos_token = resolve_cpword_bos_eos_tokens()
    encoder_tokens, encoder_padding_mask = cpword_ids_to_batch(
        cpword_ids,
        cpword_bos_token=cpword_bos_token,
        cpword_eos_token=cpword_eos_token,
        max_source_length=max_source_length,
        device=device,
        strip_cpword_special_tokens=strip_cpword_special_tokens,
    )

    top_k = args.top_k if args.top_k > 0 else None
    top_p = 1.0 if args.top_p is None else args.top_p
    autocast_dtype = resolve_autocast_dtype(args.dtype, device)

    print(f"[INFO] device             = {device}")
    print(f"[INFO] dtype              = {args.dtype} (autocast={autocast_dtype})")
    print(f"[INFO] source tokens      = {encoder_tokens.size(1)}")
    print(f"[INFO] max_target_length  = {max_target_length}")
    print(f"[INFO] repetition_penalty = {args.repetition_penalty}")

    with torch.inference_mode(), get_amp_context(autocast_dtype):
        preds = model.generate(
            encoder_input_tokens=encoder_tokens,
            encoder_padding_mask=encoder_padding_mask,
            max_length=max_target_length,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
        )

    pred_ids = trim_token_ids(
        preds[0].tolist(),
        bos_token_id=config.model.decoder_config.bos_token_id,
        eos_token_id=config.model.decoder_config.eos_token_id,
        pad_token_id=config.model.decoder_config.pad_token_id,
    )
    lmx_text = decode_tokens_to_lmx(pred_ids, tokenizer)
    if not lmx_text:
        raise RuntimeError("The model produced an empty LMX sequence.")

    if args.save_lmx:
        lmx_path = output_path.with_suffix(".lmx")
        lmx_path.parent.mkdir(parents=True, exist_ok=True)
        lmx_path.write_text(lmx_text, encoding="utf-8")
        print(f"[INFO] wrote LMX to {lmx_path}")

    try:
        xml_text = tokenizer.converter.delinearize(lmx_text)
    except Exception as exc:
        lmx_path = output_path.with_suffix(".lmx")
        lmx_path.parent.mkdir(parents=True, exist_ok=True)
        lmx_path.write_text(lmx_text, encoding="utf-8")
        raise RuntimeError(
            f"Failed to convert generated LMX to MusicXML: {exc}. Saved LMX to {lmx_path}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_text, encoding="utf-8")
    print(f"[INFO] wrote MusicXML to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe a MIDI file to MusicXML with a MIDI2Score checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--midi", type=Path, required=True, help="Input MIDI file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model weights (.pt).")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="LMX BPE tokenizer (tokenizer.json).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Seq2Seq YAML whose architecture matches the checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MusicXML path. Defaults to the MIDI path with a .musicxml suffix.",
    )
    parser.add_argument(
        "--save-lmx",
        action="store_true",
        help="Also write the intermediate LMX next to the MusicXML file.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=None,
        help="CPWord length cap, including BOS/EOS. Defaults to config data.max_source_length.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=None,
        help="Maximum generated LMX tokens. Defaults to config data.max_target_length.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling. Values <= 0 disable it.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling cutoff. Omit to disable.")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty for repeated tokens.")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_source_length is not None and args.max_source_length <= 0:
        raise ValueError("--max-source-length must be positive")
    if args.max_target_length is not None and args.max_target_length <= 0:
        raise ValueError("--max-target-length must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be a float in (0, 1]")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        transcribe(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
