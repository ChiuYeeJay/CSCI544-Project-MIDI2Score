"""MIDI2Score Seq2Seq package."""

from midi2score.data_seq2seq import (
    HuggingFaceSeq2SeqDataset,
    Seq2SeqBatch,
    Seq2SeqDataConfig,
    build_seq2seq_dataloader,
    collate_seq2seq_batch,
)

from midi2score.model_seq2seq import (
    Seq2SeqConfig,
    TransformerForConditionalGeneration,
)

from midi2score.model_decoder import (
    DecoderLanguageModelConfig,
    TransformerDecoderLM,
    SinusoidalPositionalEncoding,
)

from midi2score.train_seq2seq import (
    Seq2SeqTrainingConfig,
    run_seq2seq_training_loop,
)

__all__ = [
    # data
    "HuggingFaceSeq2SeqDataset",
    "Seq2SeqBatch",
    "Seq2SeqDataConfig",
    "build_seq2seq_dataloader",
    "collate_seq2seq_batch",

    # model
    "Seq2SeqConfig",
    "TransformerForConditionalGeneration",
    "DecoderLanguageModelConfig",
    "TransformerDecoderLM",
    "SinusoidalPositionalEncoding",

    # train
    "Seq2SeqTrainingConfig",
    "Seq2SeqTrainingResult",
    "run_seq2seq_training_loop",
]