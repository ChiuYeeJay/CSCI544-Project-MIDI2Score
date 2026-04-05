# Best Models

This directory is reserved for project-best checkpoints and related metadata.

It is intentionally tracked by git and is not covered by the current `.gitignore`.

Current recommended checkpoints:

- `rd`:
  - config: `configs/pretrain_rd_best.yaml`
  - checkpoint: `artifacts/pretrained_decoder_rd_best_best.pt`
- `full`:
  - config: `configs/pretrain_full_best.yaml`
  - checkpoint: `artifacts/research/EXP-FULL-RDREF-003_sliding160_dmodel256_ff1024_lr6e4_linearwarmup_bs8_long/best.pt`

Recommended usage:

- keep large binary checkpoints here only when you explicitly want them versioned
- otherwise use this directory for small metadata files, manifests, or manually selected exported checkpoints
