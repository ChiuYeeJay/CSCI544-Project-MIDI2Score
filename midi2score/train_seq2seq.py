from __future__ import annotations

import math
import os
import time
import copy
import shutil
from dataclasses import asdict, dataclass, field
from lightning.pytorch.callbacks import Callback
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from peft import LoraConfig, get_peft_model
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from midi2score.data_seq2seq import Seq2SeqDataConfig, build_seq2seq_dataloader
from midi2score.model_seq2seq import Seq2SeqConfig, TransformerForConditionalGeneration

@dataclass(slots=True)
class Seq2SeqTrainingConfig:
    pretrained_decoder_path: str | None = None
    batch_size: int = 8
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float | None = 1.0
    label_smoothing: float = 0.0

    scheduler: str = "none"   # none / linear / cosine
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0

    num_steps: int | None = 1000
    num_epochs: int | None = None
    max_duration_seconds: float | None = None

    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    log_every: int = 10
    val_check_interval: int | float = 50
    check_val_every_n_epoch: int | None = None
    num_eval_batches: int | None = None

    device: str = "auto"

    save_checkpoint_path: str | None = None
    save_best_checkpoint_path: str | None = None
    resume_checkpoint_path: str | None = None

    csv_log_path: str | None = None
    tensorboard_log_dir: str | None = None

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    curriculum_learning: bool = False
    curriculum_epoch_schedule: list[int] = field(default_factory=lambda: [0, 5, 15])

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.scheduler not in {"none", "linear", "cosine"}:
            raise ValueError("scheduler must be one of none/linear/cosine.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1.")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1.")
        if self.num_steps is None and self.num_epochs is None:
            raise ValueError("Either num_steps or num_epochs must be specified.")
        if self.num_steps is not None and self.num_epochs is not None:
            raise ValueError("Only one of num_steps or num_epochs can be specified.")
        if self.num_steps is not None and self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.num_epochs is not None and self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if isinstance(self.val_check_interval, int):
            if self.val_check_interval <= 0:
                raise ValueError("val_check_interval must be a positive integer.")
        elif isinstance(self.val_check_interval, float):
            if not (0.0 < self.val_check_interval <= 1.0):
                raise ValueError("Float val_check_interval must be in (0, 1].")
        else:
            raise TypeError("val_check_interval must be an int or a float.")
        if self.check_val_every_n_epoch is not None and self.check_val_every_n_epoch <= 0:
            raise ValueError("check_val_every_n_epoch must be positive.")
        if isinstance(self.val_check_interval, float) and self.check_val_every_n_epoch is None:
            raise ValueError("If val_check_interval is a fraction, check_val_every_n_epoch must be set.")
        if self.num_eval_batches is not None and self.num_eval_batches <= 0:
            raise ValueError("num_eval_batches must be positive.")
        if self.resume_checkpoint_path is not None and not Path(self.resume_checkpoint_path).exists():
            raise ValueError(f"resume_checkpoint_path does not exist: {self.resume_checkpoint_path}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Seq2SeqTrainingResult:
    device: str
    checkpoint_path: str | None
    best_checkpoint_path: str | None
    best_validation_loss: float | None
    final_step: int
    resumed_from_checkpoint: str | None
    optimizer_state_loaded: bool
    elapsed_seconds: float
    stopped_due_to_time_budget: bool
    stopped_due_to_early_stopping: bool

def load_pretrained_decoder(seq2seq_model, checkpoint_path):
    """在套用 PEFT 前呼叫此函數載入權重"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pretrained_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    model_dict = seq2seq_model.state_dict()

    loaded = []
    skipped = []

    for k, v in pretrained_dict.items():
        new_k = "model.decoder." + k

        if new_k in model_dict:
            if model_dict[new_k].shape == v.shape:
                model_dict[new_k] = v
                loaded.append(new_k)
            else:
                skipped.append((new_k, "shape mismatch"))
        else:
            skipped.append((new_k, "not found"))

    seq2seq_model.load_state_dict(model_dict, strict=False)

    print(f"✅ Pretrained Decoder Loaded: {len(loaded)}")
    if skipped:
        print(f"⚠️ Pretrained Decoder Skipped: {len(skipped)}")

    return loaded, skipped


# =========================
# Lightning Module Wrapper
# =========================
class LitSeq2Seq(TransformerForConditionalGeneration, L.LightningModule):
    def __init__(self, config: Seq2SeqConfig, training_config: Seq2SeqTrainingConfig):
        super().__init__(config)
        self.training_config = training_config

    def training_step(self, batch, batch_idx):
        loss, logits = self.forward(
            encoder_input_tokens=batch.encoder_input_tokens,
            decoder_input_tokens=batch.decoder_input_tokens,
            labels=batch.labels,
            encoder_padding_mask=batch.encoder_padding_mask,
        )

        if self.training_config.label_smoothing > 0.0:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = batch.labels.reshape(-1)
            loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                label_smoothing=self.training_config.label_smoothing,
            )

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.forward(
            encoder_input_tokens=batch.encoder_input_tokens,
            decoder_input_tokens=batch.decoder_input_tokens,
            labels=batch.labels,
            encoder_padding_mask=batch.encoder_padding_mask,
        )

        # 計算 Metrics
        flat_labels = batch.labels.reshape(-1)
        valid_mask = flat_labels.ne(-100)
        valid_labels = flat_labels[valid_mask]

        token_acc = 0.0
        top5_acc = 0.0

        if valid_labels.numel() > 0:
            flat_logits = logits.reshape(-1, logits.size(-1))[valid_mask]
            predictions = flat_logits.argmax(dim=-1)
            topk = flat_logits.topk(k=min(5, flat_logits.size(-1)), dim=-1).indices

            token_acc = predictions.eq(valid_labels).float().mean()
            top5_acc = topk.eq(valid_labels.unsqueeze(-1)).any(dim=-1).float().mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/token_acc", token_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/top5_acc", top5_acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_parameters,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        print(f"Estimated total training steps: {total_steps}")

        warmup_steps = max(1, math.ceil(total_steps * self.training_config.warmup_ratio))
        print(
            f"Warmup ratio: {self.training_config.warmup_ratio:.4f} -> "
            f"warmup steps: {warmup_steps}"
        )

        min_lr_ratio = self.training_config.min_lr_ratio
        scheduler_name = self.training_config.scheduler

        if scheduler_name == "none" and warmup_steps == 0:
            return optimizer

        if scheduler_name == "none":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif scheduler_name == "linear":
            if min_lr_ratio > 0.0:
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    lr_end=self.training_config.learning_rate * min_lr_ratio,
                    power=1.0,
                )
            else:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )
        else:
            if min_lr_ratio > 0.0:
                scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_rate=min_lr_ratio,
                )
            else:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _validate_setup(model_config: Seq2SeqConfig, data_config: Seq2SeqDataConfig) -> None:
    if model_config.encoder_config.max_length < data_config.max_source_length:
        raise ValueError("model.max_source_length must be >= data.max_source_length.")
    if model_config.decoder_config.max_length < data_config.max_target_length:
        raise ValueError("model.max_target_length must be >= data.max_target_length.")

    tokenizer_vocab = data_config.tokenizer_vocab_size()
    if tokenizer_vocab is not None:
        if tokenizer_vocab != model_config.decoder_config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({tokenizer_vocab}) != "
                f"model tgt_vocab_size ({model_config.decoder_config.vocab_size})"
            )


# =========================
# Curriculum Learning Callback
# =========================
class CurriculumLearningCallback(Callback):
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        training_config = pl_module.training_config
        epoch = trainer.current_epoch
        
        if epoch in training_config.curriculum_epoch_schedule:
            target_stage = training_config.curriculum_epoch_schedule.index(epoch)
            
            train_loader = trainer.train_dataloader
            if train_loader is not None:
                dataset = train_loader.dataset
                dataset.set_stage(target_stage)
                
                pl_module.log("curriculum/stage_index", target_stage)
                
            if trainer.is_global_zero:
                print(f"\n[Curriculum] Epoch {epoch}: Switching to stage '{target_stage}'")

# =========================
# Main Training Logic
# =========================
def run_seq2seq_training_loop(
    model_config: Seq2SeqConfig,
    data_config: Seq2SeqDataConfig,
    training_config: Seq2SeqTrainingConfig,
) -> Seq2SeqTrainingResult:
    
    _validate_setup(model_config, data_config)

    # 1. 建立 DataLoader
    train_loader = build_seq2seq_dataloader(data_config, batch_size=training_config.batch_size)
    
    val_loader = None
    has_validation = training_config.val_check_interval > 0

    if has_validation:
        val_data_config = copy.deepcopy(data_config)
        val_data_config.split = "validation"
        val_loader = build_seq2seq_dataloader(val_data_config, batch_size=training_config.batch_size, shuffle=False)

    # 2. 初始化模型
    model = LitSeq2Seq(model_config, training_config)

    # 3. 載入 Pretrain Decoder 參數（必須在掛 LoRA 之前載入）
    if training_config.pretrained_decoder_path is not None:
        load_pretrained_decoder(model, training_config.pretrained_decoder_path)

    # 4. 依設定決定是否對 Decoder 套用 PEFT LoRA
    if training_config.use_lora:
        peft_config = LoraConfig(
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            target_modules=[
                "self_attn.q_proj", 
                "self_attn.k_proj", 
                "self_attn.v_proj", 
                "self_attn.out_proj", 
                "feedforward.linear1", 
                "feedforward.linear2",
            ],
            lora_dropout=training_config.lora_dropout,
            bias="none",
        )
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)

        # 5. Unfreezing
        # 5.1 Unfreeze Encoder
        for p in model.model.encoder.parameters():
            p.requires_grad = True
            
        # 5.2 Unfreeze cross_attention in Decoder
        for name, p in model.model.decoder.named_parameters():
            if "cross_attn" in name:
                p.requires_grad = True
        
        print("\n==== TRAINABLE PARAMS ====")
        model.model.decoder.print_trainable_parameters()

    else:
        for p in model.parameters():
            p.requires_grad = True
        
        print("\n==== TRAINABLE PARAMS ====")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        ratio = 100.0 * trainable_params / max(total_params, 1)
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {total_params:,} || trainable%: {ratio:.4f}"
        )
    
    # 6. 設定 PyTorch Lightning Loggers
    loggers = []
    if training_config.tensorboard_log_dir:
        loggers.append(TensorBoardLogger(save_dir=training_config.tensorboard_log_dir, name="seq2seq"))
    if training_config.csv_log_path:
        loggers.append(CSVLogger(save_dir=Path(training_config.csv_log_path).parent, name="csv_logs"))

    # 7. 設定 Callbacks
    callbacks = [LearningRateMonitor(logging_interval='step'), DeviceStatsMonitor()]
    
    if training_config.save_best_checkpoint_path and has_validation:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(training_config.save_best_checkpoint_path).parent,
                filename="best-checkpoint-{step}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            )
        )

    if training_config.early_stopping_patience is not None and has_validation:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=training_config.early_stopping_patience,
                min_delta=training_config.early_stopping_min_delta,
                mode="min"
            )
        )
    
    if training_config.curriculum_learning:
        callbacks.append(CurriculumLearningCallback())

    # 8. 建立 Lightning Trainer
    trainer = L.Trainer(
        accelerator=training_config.device if training_config.device != "auto" else "auto",
        devices=1,
        precision=training_config.precision,
        max_steps=training_config.num_steps if training_config.num_steps is not None else -1,
        max_epochs=training_config.num_epochs,
        val_check_interval=training_config.val_check_interval,
        check_val_every_n_epoch=training_config.check_val_every_n_epoch,
        limit_val_batches=training_config.num_eval_batches if training_config.num_eval_batches else 1.0,
        gradient_clip_val=training_config.grad_clip_norm,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        log_every_n_steps=training_config.log_every,
        reload_dataloaders_every_n_epochs=1,
        max_time=None if not training_config.max_duration_seconds else timedelta(seconds=training_config.max_duration_seconds)
    )

    # 9. 開始訓練
    print("\n==== START TRAINING ====")
    start_time = time.monotonic()

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=training_config.resume_checkpoint_path
    )
    
    elapsed_seconds = time.monotonic() - start_time
    print(f"\n==== TRAINING FINISHED in {elapsed_seconds:.2f} seconds ====")

    # 10. 儲存最後的 Checkpoint (如果需要)
    if training_config.save_checkpoint_path:
        trainer.save_checkpoint(training_config.save_checkpoint_path)

    best_ckpt_path = None
    best_val_loss = None
    for cb in trainer.callbacks:
        if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint):
            best_ckpt_path = cb.best_model_path
            best_val_loss = cb.best_model_score.item() if cb.best_model_score else None

    if best_ckpt_path and training_config.save_best_checkpoint_path:
        target_path = Path(training_config.save_best_checkpoint_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(best_ckpt_path, target_path)

        cleanup_root = target_path.parent
        for dirpath, dirnames, filenames in os.walk(cleanup_root, topdown=False):
            current_dir = Path(dirpath)
            if current_dir == cleanup_root:
                continue
            if not dirnames and not filenames:
                current_dir.rmdir()

        best_ckpt_path = str(target_path)

    stopped_due_to_early_stopping = any(
        isinstance(cb, L.pytorch.callbacks.EarlyStopping) and trainer.should_stop 
        for cb in trainer.callbacks
    )

    return Seq2SeqTrainingResult(
        device=str(trainer.device_ids),
        checkpoint_path=training_config.save_checkpoint_path,
        best_checkpoint_path=best_ckpt_path,
        best_validation_loss=best_val_loss,
        final_step=trainer.global_step,
        resumed_from_checkpoint=training_config.resume_checkpoint_path,
        optimizer_state_loaded=training_config.resume_checkpoint_path is not None,
        elapsed_seconds=elapsed_seconds,
        stopped_due_to_time_budget=False,
        stopped_due_to_early_stopping=stopped_due_to_early_stopping,
    )