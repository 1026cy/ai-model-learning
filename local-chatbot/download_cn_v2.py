"""下载真正的中文开源数据集"""
import json
import time
from pathlib import Path

def main():
    from datasets import load_dataset
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("  📥 真正的中文对话数据集")
    print("=" * 50)
    
    # 尝试多个中文数据集
    cn_sources = [
        # 清华 BELLE 中文指令数据
        ("BAAI/BELLE-1.5M-2M", "belle_cn.json", "BELLE中文 (2M+, 清华)"),
        # 如果上面太大，用小一点的
        # ("silk-road/alpaca-data-gpt4-chinese", "alpaca_gpt4_cn.json", "alpaca中文"),
    ]
    
    for hf_name, filename, desc in cn_sources:
        print(f"\n📌 {desc}")
        print(f"   HF地址: {hf_name}")
        try:
            start = time.time()
            ds = load_dataset(hf_name, split="train")
            print(f"  ✓ 下载完成: {len(ds)} 条 ({time.time()-start:.1f}s)")
            
            # 只取前10000条先，避免太大
            data = [dict(row) for row in ds.select(range(min(10000, len(ds))))]
            
            # 标准化字段名: input -> instruction, output -> response
            normalized = []
            for item in data:
                norm_item = {
                    "instruction": item.get("input", item.get("instruction", "")),
                    "input": "",
                    "output": item.get("target", item.get("response", item.get("output", ""))),
                }
                normalized.append(norm_item)
            
            output_path = raw_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)
            
            size_mb = output_path.stat().st_size / (1024*1024)
            print(f"  ✓ 已保存: {output_path} ({size_mb:.1f} MB, 先取10000条)")
            
            # 验证中文
            import re
            zh_count = sum(1 for item in normalized[:500] if re.search(r'[\u4e00-\u9fff]', item['instruction'] + item['output']))
            print(f"  ✓ 中文占比: {zh_count/5:.1f}%")
            
            if normalized:
                print(f"\n  中文样例:")
                print(f"    instruction: {normalized[0]['instruction'][:100]}")
                print(f"    output: {normalized[0]['output'][:100]}")
                
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("  ✅ 完成!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
