"""
训练引擎 - 封装 Trainer / SFTTrainer
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict

logger = logging.getLogger(__name__)


class CausalLMCollator:
    """Pad causal-LM batches and mask only padding positions in their labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def create_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """
    创建 HuggingFace TrainingArguments
    
    Args:
        config: 完整配置字典
    
    Returns:
        TrainingArguments 实例
    """
    train_config = config["training"]
    eval_steps = train_config.get("eval_steps", 0) or 0
    do_eval = eval_steps > 0

    # 构建参数
    kwargs = {
        "output_dir": train_config["output_dir"],
        "per_device_train_batch_size": train_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": train_config["gradient_accumulation_steps"],
        "learning_rate": train_config["learning_rate"],
        "lr_scheduler_type": train_config["lr_scheduler_type"],
        "warmup_ratio": train_config["warmup_ratio"],
        "weight_decay": train_config["weight_decay"],
        "num_train_epochs": train_config["num_train_epochs"],
        "max_steps": train_config.get("max_steps", -1),
        "logging_steps": train_config["logging_steps"],
        "save_steps": train_config["save_steps"],
        "save_total_limit": train_config["save_total_limit"],
        "optim": train_config.get("optim", "adamw_torch"),
        "fp16": train_config.get("fp16", False) and torch.cuda.is_available(),
        "max_grad_norm": train_config.get("max_grad_norm", 0.3),
        "gradient_checkpointing": config["resources"].get("gradient_checkpointing", False),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # 数据加载
        "dataloader_num_workers": config["resources"].get("dataloader_num_workers", 2),
        "dataloader_persistent_workers": config["resources"].get("dataloader_num_workers", 2) > 0,
        "dataloader_pin_memory": True,
        # 日志
        "report_to": config["logging"].get("report_to", "none"),
        "run_name": config["logging"].get("run_name", "local-chatbot"),
        # 保存策略
        "save_strategy": config["save"].get("save_strategy", "steps"),
        "remove_unused_columns": False,  # SFTTrainer 需要保留所有列
        # 禁用 HF Hub 上传
        "push_to_hub": False,
    }

    # 评估策略：eval_steps<=0 时完全关闭验证
    eval_key = "eval_strategy" if "eval_strategy" in TrainingArguments.__dataclass_fields__ else "evaluation_strategy"
    if do_eval:
        kwargs[eval_key] = "steps"
        kwargs["eval_steps"] = eval_steps
    else:
        kwargs[eval_key] = "no"
    
    # 处理 save_format
    if config["save"].get("save_format") == "adapter":
        kwargs["save_only_model"] = True  # QLoRA 训练时只保存 adapter
    
    args = TrainingArguments(**kwargs)
    
    logger.info(f"TrainingArguments:")
    logger.info(f"  output_dir: {args.output_dir}")
    logger.info(f"  batch_size: {args.per_device_train_batch_size}")
    logger.info(f"  grad_accum: {args.gradient_accumulation_steps}")
    logger.info(f"  effective_batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  fp16: {args.fp16}")
    logger.info(f"  epochs: {args.num_train_epochs}")
    
    return args


def train_model(
    model,
    tokenizer,
    dataset: DatasetDict,
    config: Dict[str, Any],
):
    """
    执行训练
    
    Args:
        model: 模型 (已加载 + 可选 LoRA)
        tokenizer: tokenizer
        dataset: DatasetDict (包含 train 和 test/validation)
        config: 完整配置
    
    Returns:
        训练好的模型
    """
    training_args = create_training_args(config)
    do_eval = (config["training"].get("eval_steps", 0) or 0) > 0

    # Mask padding by attention_mask so EOS remains a learnable target when it is also pad_token.
    data_collator = CausalLMCollator(tokenizer)

    eval_dataset = dataset.get("test", dataset.get("validation")) if do_eval else None

    # 选择训练器类型
    trainer_type = config["training"].get("trainer_type", "sft")
    
    if trainer_type == "sft":
        # 使用 TRL 的 SFTTrainer (更好的对话格式支持)
        logger.info("使用 SFTTrainer (TRL)...")
        try:
            from trl import SFTTrainer
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                max_seq_length=config["data"]["max_seq_length"],
                packing=False,  # 对话模型通常不用 packing
            )
        except ImportError:
            logger.warning("trl 未安装，回退到标准 Trainer")
            trainer_type = "default"
    
    if trainer_type == "default":
        logger.info("使用标准 Trainer...")
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    
    # ==================== 开始训练 ====================
    logger.info("=" * 50)
    logger.info("  开始训练 🚀")
    logger.info("=" * 50)
    
    # 记录初始显存
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 固定输入尺寸下自动选最快卷积算法
        initial_mem = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"初始 GPU 显存: {initial_mem:.2f} GB")
    
    # 训练
    train_result = trainer.train()
    
    logger.info("=" * 50)
    logger.info("  训练完成 ✅")
    logger.info("=" * 50)
    logger.info(f"训练指标: {train_result.metrics}")
    
    # ==================== 保存 ====================
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config["save"].get("merge_lora_before_save") and config["lora"].get("enabled"):
        # 合并 LoRA 权重后保存完整模型
        from src.training.model_loader import merge_lora_weights
        final_path = output_dir / "merged_model"
        model_to_save = trainer.accelerator.unwrap_model(trainer.model)
        merge_lora_weights(model_to_save, str(final_path), tokenizer=tokenizer)
        logger.info(f"合并后的完整模型已保存: {final_path}")
    else:
        # 保存 adapter 或全模型
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"模型已保存: {training_args.output_dir}")
    
    # 保存训练日志
    metrics_file = output_dir / "train_metrics.json"
    import json
    with open(metrics_file, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    logger.info(f"训练指标已保存: {metrics_file}")
    
    return model, trainer
