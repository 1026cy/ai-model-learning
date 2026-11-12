"""
训练入口脚本

用法:
    python train.py                           # 使用默认配置
    python train.py --config configs/my.yaml  # 使用自定义配置
    python train.py --num_samples 100         # 快速测试 (只用100条数据)
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

# 确保项目根目录在 sys.path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="训练本地对话模型")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--num_samples", type=int, default=None, help="限制样本数量 (快速测试用)")
    parser.add_argument("--epochs", type=float, default=None, help="覆盖配置中的 epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 batch_size")
    parser.add_argument("--skip_prepare", action="store_true", help="跳过环境检查")
    parser.add_argument("--create_sample_data", action="store_true", help="创建示例自定义数据")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ==================== 环境检查 ====================
    logger.info("=" * 60)
    logger.info("  Local Chatbot - 本地对话模型训练 🤖")
    logger.info("=" * 60)
    
    # 硬件信息
    from src.utils.hardware import print_system_info, get_optimal_training_config
    
    if not args.skip_prepare:
        info = print_system_info()
    else:
        info = {"gpu": {"vram_total_gb": 6}}
    
    # ==================== 加载配置 ====================
    from src.utils.config import load_config, print_config
    
    config = load_config(args.config)
    
    # 显存自适应调整
    vram_gb = info["gpu"].get("vram_total_gb", 6)
    config = get_optimal_training_config(vram_gb, config)
    
    # 命令行覆盖
    if args.num_samples is not None:
        config["data"]["num_samples"] = args.num_samples
    if args.epochs is not None:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    
    print_config(config)
    
    # ==================== (可选) 创建示例数据 ====================
    if args.create_sample_data:
        from src.data.dataset import create_sample_custom_data
        sample_path = create_sample_custom_data(
            "data/raw/chat_data.json",
            num_samples=100,
        )
        logger.info(f"示例数据已创建: {sample_path}")
        config["data"]["dataset_name"] = "custom"
    
    # ==================== 加载模型和 Tokenizer ====================
    from src.training.model_loader import (
        load_base_model_and_tokenizer,
        apply_lora,
    )
    
    model, tokenizer = load_base_model_and_tokenizer(config)
    
    # 添加 LoRA adapter
    model = apply_lora(model, config)
    
    # ==================== 准备数据 ====================
    from src.data.dataset import prepare_dataset_for_training
    
    dataset = prepare_dataset_for_training(config, tokenizer)
    
    logger.info(f"训练集大小: {len(dataset['train'])}")
    if "test" in dataset:
        logger.info(f"验证集大小: {len(dataset['test'])}")
    
    # ==================== 开始训练 ====================
    from src.training.trainer import train_model
    
    model, trainer = train_model(model, tokenizer, dataset, config)
    
    # ==================== 完成 ====================
    logger.info("=" * 60)
    logger.info("  训练全部完成！🎉")
    logger.info(f"  模型输出目录: {config['training']['output_dir']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
