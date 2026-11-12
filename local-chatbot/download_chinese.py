"""下载中文开源对话数据集"""
import json
import time
from pathlib import Path

def main():
    from datasets import load_dataset
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("  📥 中文开源对话数据集")
    print("=" * 50)
    
    datasets_cn = [
        ("silk-road/alpaca-data-gpt4-chinese", "alpaca_gpt4_cn.json", "中文 (GPT4生成, ~48K)"),
    ]
    
    for hf_name, filename, desc in datasets_cn:
        print(f"\n📌 {desc}")
        try:
            start = time.time()
            ds = load_dataset(hf_name, split="train")
            print(f"  ✓ 下载完成: {len(ds)} 条 ({time.time()-start:.1f}s)")
            
            data = [dict(row) for row in ds]
            output_path = raw_dir / filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            size_mb = output_path.stat().st_size / (1024*1024)
            print(f"  ✓ 已保存: {output_path} ({size_mb:.1f} MB)")
            
            if data:
                print(f"\n  样例 (中文):")
                sample = data[0]
                for k in ["instruction", "input", "output"]:
                    val = sample.get(k, "")
                    if val:
                        print(f"    {k}: {val[:100]}...")
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
    
    print(f"\n{'='*50}")
    print("  ✅ 中文数据集下载完成!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
