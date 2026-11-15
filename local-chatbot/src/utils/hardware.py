"""
硬件信息检测与显存优化工具
"""
import torch
import psutil
import logging

logger = logging.getLogger(__name__)


def get_system_info():
    """获取系统硬件信息"""
    info = {
        "cpu": {},
        "memory": {},
        "gpu": {},
    }
    
    # CPU 信息
    import platform
    info["cpu"]["name"] = platform.processor()
    info["cpu"]["cores"] = psutil.cpu_count(logical=False)
    info["cpu"]["threads"] = psutil.cpu_count(logical=True)
    
    # 内存信息
    mem = psutil.virtual_memory()
    info["memory"]["total_gb"] = round(mem.total / (1024**3), 2)
    info["memory"]["available_gb"] = round(mem.available / (1024**3), 2)
    
    # GPU 信息
    if torch.cuda.is_available():
        info["gpu"]["name"] = torch.cuda.get_device_name(0)
        info["gpu"]["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info["gpu"]["vram_free_gb"] = round(torch.cuda.mem_get_info()[0] / (1024**3), 2)
        info["gpu"]["cuda_version"] = torch.version.cuda
        info["gpu"]["device_count"] = torch.cuda.device_count()
    else:
        info["gpu"]["name"] = "None (CPU only)"
        info["gpu"]["vram_total_gb"] = 0
        info["gpu"]["vram_free_gb"] = 0
        info["gpu"]["cuda_version"] = None
    
    return info


def print_system_info():
    """打印系统信息摘要"""
    info = get_system_info()
    logger.info("=" * 50)
    logger.info("  系统硬件信息")
    logger.info("=" * 50)
    logger.info(f"  CPU:  {info['cpu']['name']}")
    logger.info(f"        {info['cpu']['cores']}核 / {info['cpu']['threads']}线程")
    logger.info(f"  内存: {info['memory']['total_gb']} GB (可用 {info['memory']['available_gb']} GB)")
    if info["gpu"]["name"] != "None (CPU only)":
        logger.info(f"  GPU:  {info['gpu']['name']}")
        logger.info(f"        VRAM: {info['gpu']['vram_total_gb']} GB (可用 {info['gpu']['vram_free_gb']} GB)")
        logger.info(f"        CUDA: {info['gpu']['cuda_version']}")
    else:
        logger.warning("  GPU:  未检测到，将使用CPU训练（非常慢！）")
    logger.info("=" * 50)
    return info


def estimate_max_batch_size(vram_gb, model_params_billion, seq_length=512):
    """
    根据显存和模型大小估算最大batch_size
    
    估算公式 (保守估计):
    - FP16: 模型参数 * 2 bytes
    - 4bit量化: 模型参数 * 0.5 bytes
    - 训练激活值: batch_size * seq_length * hidden_size * 20 bytes (经验值)
    """
    # 简单启发式估算
    if vram_gb <= 4:
        bs = 1
    elif vram_gb <= 6:
        bs = 1 if model_params_billion > 1 else 2
    elif vram_gb <= 8:
        bs = 2 if model_params_billion > 1 else 4
    elif vram_gb <= 16:
        bs = 4 if model_params_billion > 3 else 8
    else:
        bs = 8 if model_params_billion > 7 else 16
    
    return bs


def cleanup_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("GPU显存已清理")


def get_optimal_training_config(vram_gb, base_config):
    """
    根据显存给出保守的默认值，但**不覆盖**用户在配置文件里显式写下的关键项
    (batch_size / gradient_accumulation_steps / use_qlora)。
    只做“缺省填充”，避免把用户调好的参数改回去。
    """
    config = base_config.copy()
    train = config.setdefault("training", {})

    # 仅当用户没有显式设置时才填充默认值
    if train.get("per_device_train_batch_size") is None:
        train["per_device_train_batch_size"] = 1 if vram_gb <= 4 else (2 if vram_gb <= 6 else 4)
    if train.get("gradient_accumulation_steps") is None:
        train["gradient_accumulation_steps"] = 16 if vram_gb <= 4 else (8 if vram_gb <= 6 else 4)

    if train.get("fp16") is None:
        train["fp16"] = vram_gb > 0
    if config.get("resources", {}).get("gradient_checkpointing") is None:
        config.setdefault("resources", {})["gradient_checkpointing"] = vram_gb <= 8

    return config
