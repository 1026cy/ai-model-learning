"""
数据集加载与预处理
支持: Alpaca, Guanaco, ShareGPT, 自定义JSON数据
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ==================== 数据格式化模板 ====================

# Alpaca 格式模板
ALPACA_PROMPT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# ShareGPT 格式模板 (对话历史)
SHAREGPT_PROMPT = """### Human:
{human_message}

### Assistant:
"""


def format_alpaca_example(example: Dict[str, Any]) -> str:
    """
    将 Alpaca 格式数据转为 prompt
    
    输入格式:
        {
            "instruction": "...",
            "input": "...",        (可选)
            "output": "..."
        }
    
    输出: prompt + response
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    response = example.get("output", "")
    
    prompt = ALPACA_PROMPT.format(
        instruction=instruction,
        input=input_text if input_text else "(无附加输入)",
    )
    
    return prompt + response


def format_sharegpt_example(example: Dict[str, Any]) -> str:
    """
    将 ShareGPT 格式数据转为 prompt
    
    输入格式 (conversations是一个列表):
        {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！有什么可以帮你？"},
                {"from": "human", "value": "介绍一下Python"}
            ]
        }
    
    输出: 多轮对话格式
    """
    conversations = example.get("conversations", [])
    formatted_parts = []
    
    for msg in conversations:
        role = msg.get("from", "").lower()
        content = msg.get("value", "")
        
        if role in ["human", "user"]:
            formatted_parts.append(f"### Human:\n{content}\n")
        elif role in ["gpt", "assistant", "bot"]:
            formatted_parts.append(f"### Assistant:\n{content}\n")
    
    return "\n".join(formatted_parts)


# ==================== 数据集加载 ====================

def load_alpaca_dataset(num_samples: Optional[int] = None) -> Dataset:
    """
    加载 Alpaca 数据集 (斯坦福开源)
    ~52K 英文指令数据
    """
    logger.info("加载 Alpaca 数据集...")
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        logger.warning(f"从HuggingFace加载失败 ({e}), 尝试加载 alpaca-cleaned...")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    logger.info(f"Alpaca 数据集: {len(dataset)} 条")
    return dataset


def load_custom_json_dataset(json_path: str, num_samples: Optional[int] = None) -> Dataset:
    """
    加载自定义 JSON 格式数据集
    
    支持两种格式:
    1. Alpaca 格式: [{"instruction": "...", "input": "...", "output": "..."}]
    2. ShareGPT 格式: [{"conversations": [{"from": "human", "value": "..."}]}]
    """
    logger.info(f"加载自定义数据集: {json_path}")
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if num_samples and num_samples < len(data):
        data = data[:num_samples]
    
    dataset = Dataset.from_list(data)
    logger.info(f"自定义数据集: {len(dataset)} 条")
    return dataset


def load_training_data(config: Dict[str, Any]) -> Dataset:
    """
    根据配置加载训练数据
    """
    data_config = config["data"]
    dataset_name = data_config["dataset_name"]
    num_samples = data_config.get("num_samples")
    
    if dataset_name == "alpaca":
        dataset = load_alpaca_dataset(num_samples)
    elif dataset_name == "custom":
        dataset = load_custom_json_dataset(data_config["custom_data_path"], num_samples)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return dataset


# ==================== 数据预处理 ====================

def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    format_func,
    max_seq_length: int = 512,
    num_proc: int = 4,
) -> Dataset:
    """
    对数据集进行格式化和分词
    
    Args:
        dataset: 原始数据集
        tokenizer: HuggingFace tokenizer
        format_func: 将样本转为文本的函数
        max_seq_length: 最大序列长度
        num_proc: 多进程处理数
    
    Returns:
        分词后的数据集，包含 input_ids 和 labels
    """
    logger.info(f"预处理数据集 (max_seq_length={max_seq_length})...")
    
    # 检查 tokenizer 是否有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"设置 pad_token = eos_token: {tokenizer.pad_token}")
    
    def preprocess_function(examples):
        # 将每条样本格式化为文本
        texts = [format_func(ex) for ex in _dict_to_list_of_dicts(examples)]
        
        # 分词
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # 设置 labels (复制 input_ids，训练时会自动处理 shift)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    logger.info(f"分词完成: {len(tokenized_dataset)} 条")
    return tokenized_dataset


def _dict_to_list_of_dicts(batch: Dict[str, List]) -> List[Dict]:
    """
    将 batch 格式 (dict of lists) 转为 list of dicts
    """
    keys = batch.keys()
    length = len(next(iter(batch.values())))
    return [{k: batch[k][i] for k in keys} for i in range(length)]


def prepare_dataset_for_training(config: Dict[str, Any], tokenizer) -> DatasetDict:
    """
    完整的数据准备流程: 加载 -> 格式化 -> 分词 -> 划分
    
    Returns:
        DatasetDict 包含 train 和 validation
    """
    # 加载原始数据
    dataset = load_training_data(config)
    
    # 选择格式化函数
    data_config = config["data"]
    dataset_name = data_config["dataset_name"]
    
    if dataset_name in ["alpaca", "custom"]:
        # 检查数据格式
        sample = dataset[0]
        if "conversations" in sample:
            format_func = format_sharegpt_example
        else:
            format_func = format_alpaca_example
    else:
        format_func = format_alpaca_example
    
    # 分词
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        format_func,
        max_seq_length=data_config["max_seq_length"],
        num_proc=config["resources"]["preprocessing_num_workers"],
    )
    
    # 划分训练集和验证集
    split_ratio = data_config.get("train_test_split", 0.1)
    if split_ratio > 0:
        splits = tokenized_dataset.train_test_split(
            test_size=split_ratio,
            shuffle=True,
            seed=42,
        )
        logger.info(f"数据集划分: train={len(splits['train'])}, val={len(splits['test'])}")
        return splits
    else:
        return DatasetDict({"train": tokenized_dataset, "test": tokenized_dataset})


# ==================== 辅助函数 ====================

def create_sample_custom_data(output_path: str, num_samples: int = 50):
    """
    创建示例自定义数据集，方便用户快速测试
    
    生成简单的中文/英文对话数据
    """
    import random
    
    samples = []
    templates = [
        # 英文模板
        {"instruction": "Explain {topic} in simple terms.", "input": "", "output": "Let me break this down simply. {topic} is..."},
        {"instruction": "Write a short story about {topic}.", "input": "", "output": "Once upon a time, there was..."},
        {"instruction": "What are the benefits of {topic}?", "input": "", "output": "Here are some key benefits of {topic}:"},
        # 中文模板  
        {"instruction": "用简单的话解释{topic}。", "input": "", "output": "让我简单来说，{topic}是..."},
        {"instruction": "写一个关于{topic}的小故事。", "input": "", "output": "从前，有一个..."},
        {"instruction": "{topic}有哪些好处？", "input": "", "output": "{topic}的主要好处包括："},
    ]
    
    topics_en = ["Python", "machine learning", "space exploration", "cooking", "music", "travel", "history"]
    topics_zh = ["人工智能", "编程", "旅行", "美食", "音乐", "历史", "运动"]
    
    for i in range(num_samples):
        tpl = random.choice(templates)
        topics = topics_en if "in simple terms" in tpl["instruction"] or "short story" in tpl["instruction"] or "benefits" in tpl["instruction"] else topics_zh
        topic = random.choice(topics)
        
        sample = {
            "instruction": tpl["instruction"].format(topic=topic),
            "input": tpl["input"],
            "output": tpl["output"].format(topic=topic),
        }
        samples.append(sample)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已生成示例数据: {output_path} ({num_samples} 条)")
    return str(output_path)
