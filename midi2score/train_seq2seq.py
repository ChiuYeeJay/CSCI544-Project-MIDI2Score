from __future__ import annotations

import time
import copy
from dataclasses import asdict, dataclass, field
from lightning.pytorch.callbacks import Callback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from peft import LoraConfig, get_peft_model

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
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    num_steps: int = 1000
    max_duration_seconds: float | None = None

    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    log_every: int = 10
    eval_every: int = 0
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
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.early_stopping_patience is not None and self.eval_every == 0:
            raise ValueError("early_stopping_patience requires eval_every > 0.")
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
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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

        if self.training_config.scheduler == "none" and self.training_config.warmup_steps == 0:
            return optimizer

        total_steps = self.training_config.num_steps
        warmup_steps = self.training_config.warmup_steps
        min_lr_ratio = self.training_config.min_lr_ratio
        scheduler_name = self.training_config.scheduler

        def lr_lambda(step: int) -> float:
            current_step = max(step, 1)

            if warmup_steps > 0 and current_step <= warmup_steps:
                return current_step / warmup_steps

            if scheduler_name == "none":
                return 1.0

            decay_start = max(warmup_steps, 1)
            decay_steps = max(total_steps - decay_start, 1)
            progress = min(max((current_step - decay_start) / decay_steps, 0.0), 1.0)

            if scheduler_name == "linear":
                return 1.0 - progress * (1.0 - min_lr_ratio)

            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

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

    if training_config.eval_every > 0:
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
            target_modules="all-linear", # 自動抓取 Decoder 中所有的 nn.Linear
            lora_dropout=training_config.lora_dropout,
            bias="none",
        )
        # 這裡直接覆蓋 base model 內的 decoder 為 PeftModel
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)

        # 5. 設定 Freezing 邏輯
        # PEFT 預設會將原本的 Decoder base_model 凍結，只開啟 lora 的 gradient
        # 但我們需要手動確保 Encoder 的狀態符合需求
        for p in model.model.encoder.parameters():
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
    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    if training_config.save_best_checkpoint_path and training_config.eval_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(training_config.save_best_checkpoint_path).parent,
                filename="best-checkpoint-{step}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            )
        )

    if training_config.early_stopping_patience is not None:
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
    val_check_interval = training_config.eval_every if training_config.eval_every > 0 else None
    
    trainer = L.Trainer(
        accelerator=training_config.device if training_config.device != "auto" else "auto",
        devices=1,
        precision=training_config.precision,
        max_steps=training_config.num_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=training_config.num_eval_batches if training_config.num_eval_batches else 1.0,
        gradient_clip_val=training_config.grad_clip_norm,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        log_every_n_steps=training_config.log_every,
        reload_dataloaders_every_n_epochs=1,
        max_time=None if not training_config.max_duration_seconds else f"00:00:00:{int(training_config.max_duration_seconds)}"
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