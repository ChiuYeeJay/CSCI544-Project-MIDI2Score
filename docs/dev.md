# MIDI2Score Seq2Seq Dev Notes

Last updated: 2026-04-05

---

# 1. Current Scope

This branch focuses on **Seq2Seq training (MIDI → LMX)** using:

* pretrained decoder (from decoder pretraining stage)
* encoder training + LoRA fine-tuning on decoder

Active goal:

* train a seq2seq model that maps MIDI token sequences → LMX token sequences

Not in scope:

* decoder pretraining (already completed)
* tokenizer training
* raw MIDI / XML parsing

---

# 2. Data Setup

## 2.1 Required Dataset

The pipeline expects a **HuggingFace DatasetDict saved on disk**:

```text
DATA/huggingface_seq2seq_rd/
  dataset_dict.json
  training/
  validation/
  test/
```

Each sample must contain:

```python
{
  "midi_clean_ids": List[List[int]]
  "midi_light_ids": List[List[int]]
  "midi_heavy_ids": List[List[int]]
  "lmx_ids": List[int]
}
```

For structure-aware truncation, generate a second dataset from the tokenized one:

```bash
python hf_dataset/hf_dataset_seq2seq_truncate.py \
  --input-path DATA/huggingface_seq2seq_rd \
  --output-path DATA/huggingface_seq2seq_truncated \
  --tokenizer-path DATA/tokenizer_rd.json \
  --max-source-length 1024 \
  --max-target-length 1024 \
  --lookahead-tokens 0
```

The truncated dataset keeps `lmx_ids` and `midi_*_ids`, and adds per-noise metadata fields such as:

```python
{
  "source_length_clean": int,
  "source_length_light": int,
  "source_length_heavy": int,
  "target_length_clean": int,
  "target_length_light": int,
  "target_length_heavy": int,
  "lmx_cutoff_clean": int,
  "lmx_cutoff_light": int,
  "lmx_cutoff_heavy": int,
}
```

---

## 2.2 Tokenizer

Path:

```text
DATA/tokenizer_rd.json
```

Must match:

* vocab size = 5000
* PAD = 0
* BOS = 1
* EOS = 2

⚠️ The system will automatically check:

* tokenizer vocab == model vocab 

---

## 2.3 Pretrained Decoder

Required:

```text
best_models/rd/best.pt
```

This checkpoint is loaded into:

```python
model.decoder
```

via weight mapping logic 

---

# 3. Pipeline Overview

Full training pipeline:

```text
Dataset (HF disk)
    ↓
Seq2SeqDataConfig
    ↓
HuggingFaceSeq2SeqDataset
    ↓
DataLoader + collate
    ↓
TransformerSeq2Seq
    ↓
Training Loop
    ↓
Checkpoint / Logs
```

---

## 3.1 Data Pipeline

Implemented in:

* `data_seq2seq.py`

Key features:

* random cropping (training)
* optional sliding window
* ratio-based weak alignment (midi ↔ lmx)
* EOS appended to decoder sequence
* padding + masking handled in collate



---

## 3.2 Model

Implemented in:

* `model_seq2seq.py`

Structure:

```text
Encoder:
  CPWordEmbedding (7-token structure)
  TransformerEncoder

Decoder:
  TransformerDecoderLM
  causal masking

Full model:
  TransformerSeq2Seq
```



---

## 3.3 Training Strategy

Implemented in:

* `train_seq2seq.py`

Key design:

### ✅ Pretrained decoder loading

* decoder weights injected into seq2seq model

### ✅ LoRA applied to decoder

* only low-rank adapters are trainable

### ✅ Freeze strategy

```text
Encoder → trainable
Decoder → frozen
LoRA → trainable
```

### ✅ Loss

```text
CrossEntropy (ignore_index = -100)
```

---

# 4. Configuration System 

This project is **fully config-driven**.

👉 All experiments are controlled by modifying:

```text
configs/seq2seq_baseline.yaml
```



---

# 4.1 Overall Structure

The config is divided into three sections:

```yaml
model:
data:
training:
```

These are parsed into:

```python
Seq2SeqConfig
Seq2SeqDataConfig
Seq2SeqTrainingConfig
```

---

# 4.2 How To Modify Config 

All experiments should be done by:

```bash
python run_seq2seq.py \
  --config configs/seq2seq_baseline.yaml \
  --experiment-id your-exp \
  --set training.learning_rate=0.0003 \
  --set model.d_model=512
```


---

# 4.3 Model Parameters (Architecture)

These control model capacity.

### 🔹 Core Parameters

| Parameter            | Meaning         | Effect                     |
| -------------------- | --------------- | -------------------------- |
| `d_model`            | hidden size     | ↑ model capacity, ↑ memory |
| `nhead`              | attention heads | must divide d_model        |
| `num_encoder_layers` | encoder depth   | ↑ representation power     |
| `num_decoder_layers` | decoder depth   | ↑ generation capacity      |
| `dim_feedforward`    | FFN size        | ↑ non-linearity            |

---

### 🔹 Sequence Length

```yaml
max_source_length
max_target_length
```


---

### 🔹 CPWord Embedding

```yaml
src_vocab_size_list
src_embedding_size_list
```


---

# 4.4 Data Parameters 



### 🔹 Dataset Path

```yaml
dataset_path
```

Must point to:

```text
DATA/huggingface_seq2seq_rd
```

---

### 🔹 Cropping Strategy

```yaml
random_crop: true
```


---

### 🔹 Sliding Window

```yaml
sliding_window_stride: null
```

Effect:

| Value | Behavior                     |
| ----- | ---------------------------- |
| null  | one sample per piece         |
| int   | multiple overlapping samples |

👉 Use when sequences are long

---

### 🔹 Length Bucketing

```yaml
length_bucketing: false
```

Effect:

* True → more efficient batching
* False → simpler training

---

### 🔹 Tokenizer Check

```yaml
tokenizer_path
```

Used to verify:

```text
vocab_size == model.tgt_vocab_size
```

---

# 4.5 Training Parameters 


## 🔹 Learning

```yaml
learning_rate
```

---

## 🔹 Batch Size

```yaml
batch_size
```

---

## 🔹 Training Length

```yaml
num_steps
```

Defines total training steps

---

## 🔹 Evaluation

```yaml
eval_every
num_eval_batches
```

Effect:

* frequent eval → better tracking
* but slower training

---

## 🔹 Scheduler

```yaml
scheduler: linear
warmup_steps: 100
min_lr_ratio: 0.1
```

Effect:

* warmup stabilizes early training
* decay improves convergence

---

## 🔹 Regularization

```yaml
weight_decay
label_smoothing
```

---

## 🔹 Gradient Control

```yaml
grad_clip_norm
```

Prevents gradient explosion

---

## 🔹 Device

```yaml
device: auto
```

Auto-select:

* CUDA
* MPS
* CPU

---

# 4.6 Decoder Fine-tuning Strategy 


```yaml
pretrained_decoder_path: best_models/rd/best.pt
```

Training behavior:

```text
Encoder → trainable
Decoder → frozen
LoRA → trainable
```


---

### 🔹 Freeze Control

```yaml
freeze_encoder
freeze_decoder
```

Default:

```yaml
freeze_encoder: false
freeze_decoder: false
```

BUT actual logic:

* decoder weights frozen manually
* LoRA is trainable


---

# 4.7 Logging & Outputs

```yaml
csv_log_path
tensorboard_log_dir
```

Outputs:

* CSV metrics
* TensorBoard logs

---

# 4.8 Checkpoint Control

```yaml
save_checkpoint_path
save_best_checkpoint_path
resume_checkpoint_path
```

Supports:

* saving latest model
* saving best validation model
* resuming training


---

# 5. How To Run

## 5.1 Environment Setup

Recommended:

```bash
uv sync --dev
```

Alternative:

```bash
pip install torch datasets pyyaml tensorboard
```

---

## 5.2 Minimal Debug Run (Recommended First)

```bash
python run_seq2seq.py --config configs/seq2seq_baseline.yaml
```

For quick test, modify config:

```yaml
training:
  num_steps: 10
  batch_size: 1
```

---

## 5.3 Full Training

```bash
python run_seq2seq.py \
  --config configs/seq2seq_baseline.yaml
```

---

## 5.4 Managed Experiment Mode

```bash
python run_seq2seq.py \
  --config configs/seq2seq_baseline.yaml \
  --experiment-id test-run \
  --set training.num_steps=1000 \
  --set model.d_model=256
```



---

# 6. Outputs

After running, outputs include:

## 6.1 Checkpoints

```text
artifacts/seq2seq/latest.pt
artifacts/seq2seq/best.pt
```

---

## 6.2 Logs

```text
logs/seq2seq.csv
logs/tensorboard/seq2seq/
```

---

## 6.3 Research Mode Outputs

```text
configs/research/<experiment_id>.yaml
artifacts/research/<experiment_id>/
logs/research/<experiment_id>.csv
```



---



