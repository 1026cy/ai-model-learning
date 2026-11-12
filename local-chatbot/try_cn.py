"""尝试下载真正的中文对话数据集 (逐个尝试)"""
import json
import time
from pathlib import Path

def main():
    from datasets import load_dataset
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 中文数据集候选 (格式: (hf_name, output_file, 描述))
    candidates = [
        # 阿里巴巴 DAMO 中文指令数据
        ("damo-nlp/COIGC", "coigc_cn.json", "COIGC 阿里中文 (可能需要申请)"),
        # 直接可用的中文数据
        ("OpenCSG/SmolTalk-Chinese-200k", "smoltalk_cn.json", "OpenCSG 中文指令 (200K)"),
        # DeepSeek R1 蒸馏中文
        ("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "deepseek_r1_cn.json", "DeepSeek R1 中文蒸馏 (110K)"),
        # 兜底: 用 alpaca + 机器翻译(跳过)
    ]
    
    print("=" * 50)
    print("  📥 中文开源数据集下载")
    print("=" * 50)
    
    for hf_name, filename, desc in candidates:
        print(f"\n--- 尝试: {desc} ---")
        print(f"    {hf_name}")
        
        try:
            start = time.time()
            ds = load_dataset(hf_name, split="train")
            print(f"    ✓ 成功! {len(ds)} 条 ({time.time()-start:.1f}s)")
            
            data = [dict(row) for row in ds.select(range(min(10000, len(ds))))]
            output_path = raw_dir / filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            size_mb = output_path.stat().st_size / (1024*1024)
            print(f"    ✓ 已保存: {output_path} ({size_mb:.1f} MB)")
            
            # 打印字段结构
            if data:
                print(f"    字段: {list(data[0].keys())}")
                print(f"    样例: {json.dumps(data[0], ensure_ascii=False)[:200]}...")
            
            break  # 成功一个就停
            
        except Exception as e:
            print(f"    ✗ 失败: {str(e)[:100]}")
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    main()
