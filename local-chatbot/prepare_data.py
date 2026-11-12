"""
数据准备脚本 - 下载/生成/转换训练数据

用法:
    python prepare_data.py download-alpaca          # 下载 Alpaca 数据集到本地
    python prepare_data.py create-sample            # 生成示例自定义数据
    python prepare_data.py create-sample --num 500  # 生成500条示例数据
    python prepare_data.py convert --input data/raw/xxx.json --format sharegpt  # 转换格式
    python prepare_data.py analyze --input data/raw/chat_data.json  # 分析数据统计
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_data")

project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))


def download_alpaca(output_dir: str = "data/raw"):
    """下载 Alpaca 数据集并保存为本地 JSON"""
    from datasets import load_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("下载 Alpaca 数据集 (yahma/alpaca-cleaned)...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    output_path = output_dir / "alpaca_cleaned.json"
    data = [dict(row) for row in dataset]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"下载完成: {output_path} ({len(data)} 条)")
    return str(output_path)


def create_sample_data(output_path: str = "data/raw/chat_data.json", num_samples: int = 100):
    """创建示例自定义数据"""
    from src.data.dataset import create_sample_custom_data
    return create_sample_custom_data(output_path, num_samples)


def convert_sharegpt_to_alpaca(input_path: str, output_path: str):
    """
    将 ShareGPT 格式数据转为 Alpaca 格式
    
    ShareGPT 格式:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    
    Alpaca 格式:
        {"instruction": "...", "input": "", "output": "..."}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"加载 ShareGPT 数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        
        # 取最后一轮 human -> gpt 作为 instruction -> response
        human_msgs = [c["value"] for c in convs if c.get("from") in ["human", "user"]]
        gpt_msgs = [c["value"] for c in convs if c.get("from") in ["gpt", "assistant", "bot"]]
        
        if human_msgs and gpt_msgs:
            sample = {
                "instruction": human_msgs[-1],  # 最后一条 human 消息
                "input": "",
                "output": gpt_msgs[-1],  # 对应的 gpt 回复
            }
            converted.append(sample)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    logger.info(f"转换完成: {len(converted)} 条 -> {output_path}")
    return str(output_path)


def analyze_dataset(input_path: str):
    """分析数据集统计信息"""
    input_path = Path(input_path)
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info("=" * 50)
    logger.info(f"  数据集分析: {input_path}")
    logger.info("=" * 50)
    logger.info(f"  总条数: {len(data)}")
    
    # 判断格式
    sample = data[0]
    if "instruction" in sample and "output" in sample:
        logger.info("  格式: Alpaca")
        lengths = []
        for item in data:
            text = item.get("instruction", "") + " " + item.get("input", "") + " " + item.get("output", "")
            lengths.append(len(text))
        
        logger.info(f"  平均文本长度: {sum(lengths)/len(lengths):.0f} 字符")
        logger.info(f"  最小/最大长度: {min(lengths)} / {max(lengths)} 字符")
        
        # 语言检测 (简单启发式)
        zh_count = sum(1 for l in lengths if False)  # placeholder
        import re
        has_chinese = sum(1 for item in data if re.search(r'[\u4e00-\u9fff]', item.get("instruction", "") + item.get("output", "")))
        logger.info(f"  含中文样本: {has_chinese}/{len(data)} ({has_chinese/len(data)*100:.1f}%)")
        
    elif "conversations" in sample:
        logger.info("  格式: ShareGPT")
        conv_lens = [len(item["conversations"]) for item in data]
        logger.info(f"  平均对话轮数: {sum(conv_lens)/len(conv_lens):.1f}")
        logger.info(f"  最小/最大轮数: {min(conv_lens)} / {max(conv_lens)}")
    
    logger.info("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="数据准备工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # download-alpaca
    p_download = subparsers.add_parser("download-alpaca", help="下载 Alpaca 数据集")
    p_download.add_argument("--output", type=str, default="data/raw", help="输出目录")
    
    # create-sample
    p_sample = subparsers.add_parser("create-sample", help="创建示例数据")
    p_sample.add_argument("--output", type=str, default="data/raw/chat_data.json", help="输出路径")
    p_sample.add_argument("--num", type=int, default=100, help="样本数量")
    
    # convert
    p_convert = subparsers.add_parser("convert", help="格式转换")
    p_convert.add_argument("--input", type=str, required=True, help="输入文件")
    p_convert.add_argument("--output", type=str, required=True, help="输出文件")
    p_convert.add_argument("--format", type=str, required=True, choices=["sharegpt2alpaca"], help="转换方向")
    
    # analyze
    p_analyze = subparsers.add_parser("analyze", help="分析数据集")
    p_analyze.add_argument("--input", type=str, required=True, help="数据文件")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.command == "download-alpaca":
        download_alpaca(args.output)
    elif args.command == "create-sample":
        create_sample_data(args.output, args.num)
    elif args.command == "convert":
        if args.format == "sharegpt2alpaca":
            convert_sharegpt_to_alpaca(args.input, args.output)
    elif args.command == "analyze":
        analyze_dataset(args.input)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
