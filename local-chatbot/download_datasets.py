"""
一键下载开源对话数据集
"""
import json
import sys
import time
from pathlib import Path

def download_dataset(dataset_name, output_path, split="train"):
    """下载数据集并保存为本地JSON"""
    from datasets import load_dataset
    
    print(f"\n{'='*50}")
    print(f"  下载: {dataset_name}")
    print(f"{'='*50}")
    
    start = time.time()
    ds = load_dataset(dataset_name, split=split)
    print(f"  ✓ 下载完成: {len(ds)} 条 ({time.time()-start:.1f}s)")
    
    # 转成普通list保存
    data = [dict(row) for row in ds]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"  ✓ 已保存: {output_path} ({size_mb:.1f} MB)")
    
    # 打印样例
    print(f"\n  样例数据 (第1条):")
    print(f"  {json.dumps(data[0], ensure_ascii=False)[:300]}...")
    
    return data


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = [
        # (name_on_hf, local_filename, description)
        ("yahma/alpaca-cleaned", "alpaca_cleaned.json", "英文指令数据 (51.7K, 斯坦福)"),
    ]
    
    # 可选: 中文数据集（需要联网，部分可能较慢）
    chinese_datasets = [
        # ("silk-road/alpaca-data-gpt4-chinese", "alpaca_gpt4_cn.json", "中文指令数据 (48K)"),
        # ("OpenAssistant/oasst1", "oasst1.json", "多语言对话数据 (60K+)"),
    ]
    
    print("=" * 50)
    print("  📥 开源对话数据集下载")
    print("=" * 50)
    
    all_data = {}
    
    # 下载主要数据集
    for hf_name, filename, desc in datasets_to_download:
        print(f"\n📌 {desc}")
        try:
            data = download_dataset(hf_name, raw_dir / filename)
            all_data[filename] = data
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
    
    print(f"\n{'='*50}")
    print("  ✅ 下载完成!")
    print(f"{'='*50}")
    print(f"\n已下载的数据集:")
    for name, data in all_data.items():
        print(f"  • {name}: {len(data)} 条")
    print(f"\n数据保存在: {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
