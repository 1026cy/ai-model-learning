"""
配置加载工具
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径，默认为 configs/training_config.yaml
    
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认路径
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "training_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置到文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def print_config(config: Dict[str, Any]):
    """打印配置摘要"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("  训练配置")
    logger.info("=" * 50)
    logger.info(f"  基础模型: {config['model']['base_model_name']}")
    logger.info(f"  数据集:   {config['data']['dataset_name']}")
    logger.info(f"  LoRA:     {'启用' if config['lora']['enabled'] else '禁用'}")
    if config['lora']['enabled']:
        logger.info(f"  QLoRA:    {'启用' if config['lora']['use_qlora'] else '禁用'}")
    logger.info(f"  Batch:    {config['training']['per_device_train_batch_size']} x {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Epochs:   {config['training']['num_train_epochs']}")
    logger.info(f"  LR:       {config['training']['learning_rate']}")
    logger.info(f"  输出目录: {config['training']['output_dir']}")
    logger.info("=" * 50)
