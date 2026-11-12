"""
模型加载与 LoRA / QLoRA 配置
"""
import logging
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    QLoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    TaskType,
)

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(config: Dict[str, Any]):
    """
    加载基础模型和 tokenizer
    
    根据配置决定:
    - 是否使用 4bit 量化 (QLoRA)
    - 是否使用 fp16
    - 梯度检查点等
    
    Returns:
        (model, tokenizer)
    """
    model_config = config["model"]
    lora_config = config["lora"]
    
    model_name = model_config["base_model_name"]
    logger.info(f"加载基础模型: {model_name}")
    
    # ==================== 配置量化 (QLoRA) ====================
    quantization_config = None
    
    if lora_config.get("enabled") and lora_config.get("use_qlora"):
        q_config = lora_config["quantization"]
        logger.info(f"启用 QLoRA (4bit量化, type={q_config['bnb_4bit_quant_type']})")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=q_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=q_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=q_config.get("bnb_4bit_use_double_quant", True),
        )
    
    # ==================== 加载 Tokenizer ====================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=model_config.get("use_fast_tokenizer", True),
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info(f"设置 pad_token = eos_token: {tokenizer.pad_token}")
    
    # ==================== 加载模型 ====================
    torch_dtype = torch.float16 if config["training"].get("fp16") else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype if not quantization_config else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    
    logger.info(f"模型加载完成: {model.dtype}, device={model.device}")
    
    # ==================== 准备 QLoRA 训练 ====================
    if quantization_config is not None:
        # 冻结 base model 参数，只训练 LoRA adapter
        model = prepare_model_for_kbit_training(model)
        logger.info("模型已准备好 QLoRA 训练 (base params frozen)")
    
    # ==================== 梯度检查点 ====================
    if config["resources"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # gradient checkpointing 与 cache 不兼容
        logger.info("启用梯度检查点")
    
    return model, tokenizer


def apply_lora(model, config: Dict[str, Any]):
    """
    为模型添加 LoRA adapter
    
    Args:
        model: 基础模型
        config: 完整配置字典
    
    Returns:
        带 LoRA adapter 的模型
    """
    lora_cfg = config["lora"]
    
    if not lora_cfg.get("enabled"):
        logger.info("LoRA 未启用，使用全量训练")
        return model
    
    logger.info("配置 LoRA...")
    
    # 构建 LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    
    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def merge_lora_weights(model, output_path: str):
    """
    合并 LoRA adapter 权重到基础模型并保存
    
    Args:
        model: 带 LoRA 的模型
        output_path: 保存路径
    """
    from peft import PeftModel
    
    if isinstance(model, PeftModel):
        logger.info("合并 LoRA 权重到基础模型...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        logger.info(f"合并后的模型已保存: {output_path}")
    else:
        # 检查是否是 prepare_model_for_kbit_training 后的模型
        if hasattr(model, "base_model"):
            logger.info("模型已被 prepare_model_for_kbit_training 处理，尝试合并...")
        model.save_pretrained(output_path)


def load_peft_model(
    base_model_name: str,
    adapter_path: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
):
    """
    加载基础模型 + 训练好的 LoRA adapter
    
    用于推理时加载
    
    Args:
        base_model_name: 基础模型名称或路径
        adapter_path: LoRA adapter 保存路径
        quantization_config: 可选的量化配置
    
    Returns:
        带 LoRA 的模型
    """
    from peft import PeftModel
    
    logger.info(f"加载推理模型: base={base_model_name}, adapter={adapter_path}")
    
    # 加载基础模型 (推理时可以不用量化，或者用更轻量的量化)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model
