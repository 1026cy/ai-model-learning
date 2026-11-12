"""
推理入口脚本 - 与训练好的模型对话

用法:
    python inference.py                                    # 交互模式
    python inference.py --model outputs/checkpoints       # 指定模型路径
    python inference.py --prompt "你好"                    # 单轮对话
    python inference.py --web                              # 启动简单Web界面
"""
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference")

project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="本地对话模型推理")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/merged_model",
                       help="训练好的模型路径")
    parser.add_argument("--base_model", type=str, default=None,
                       help="基础模型名称 (用于加载 LoRA adapter)")
    parser.add_argument("--adapter", type=str, default=None,
                       help="LoRA adapter 路径 (如果未合并)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="单轮对话输入")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="采样温度")
    parser.add_argument("--web", action="store_true",
                       help="启动 Web 对话界面")
    parser.add_argument("--use_4bit", action="store_true",
                       help="使用 4bit 量化推理 (省显存)")
    return parser.parse_args()


def load_model_for_inference(args):
    """加载模型用于推理"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_path = Path(args.model)
    
    # 判断是合并后的模型还是 adapter
    is_merged = model_path.exists() and not (model_path / "adapter_config.json").exists()
    
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        quantization_config = None
    
    if args.adapter and args.base_model:
        # 方式1: 分别指定 base model 和 adapter
        logger.info(f"加载 base model: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        logger.info(f"加载 adapter: {args.adapter}")
    elif model_path.exists():
        # 方式2: 直接加载 (可能是合并后的或 adapter 目录)
        if (model_path / "adapter_config.json").exists() and args.base_model:
            # 这是 adapter 目录
            logger.info(f"加载 adapter: {model_path}, base: {args.base_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(model_path))
        else:
            # 这是完整模型 (可能合并过)
            logger.info(f"加载完整模型: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    elif args.base_model:
        # 方式3: 只有基础模型 (未训练的)
        logger.info(f"加载基础模型: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        raise FileNotFoundError(
            f"找不到模型: {model_path}\n"
            "请先训练模型，或使用 --base_model 指定基础模型"
        )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, history=None):
    """生成回复"""
    import torch
    
    # 构建输入
    if history is None:
        # 单轮对话
        formatted_prompt = f"### Human:\n{prompt}\n\n### Assistant:\n"
    else:
        # 多轮对话
        parts = []
        for i, msg in enumerate(history):
            role = "### Human" if i % 2 == 0 else "### Assistant"
            parts.append(f"{role}:\n{msg}")
        parts.append(f"### Human:\n{prompt}")
        parts.append("### Assistant:\n")
        formatted_prompt = "\n\n".join(parts)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=temperature > 0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码并提取新生成的部分
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 只取 Assistant 的回复部分
    if "### Assistant:" in full_response:
        response = full_response.split("### Assistant:")[-1].strip()
        # 如果后面又出现 Human，截断
        if "### Human:" in response:
            response = response.split("### Human:")[0].strip()
    else:
        response = full_response
    
    return response


def chat_loop(model, tokenizer, max_new_tokens=256, temperature=0.7):
    """交互对话循环"""
    print("\n" + "=" * 50)
    print("  Local Chatbot 对话模式 💬")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'clear' 清除对话历史")
    print("=" * 50 + "\n")
    
    history = []
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 再见！")
            break
        if user_input.lower() == "clear":
            history = []
            print("🗑️  对话历史已清除")
            continue
        
        print("🤖 模型: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            history=history,
        )
        print(response)
        
        # 更新历史
        history.append(user_input)
        history.append(response)


def start_web_ui(model, tokenizer, max_new_tokens=256, temperature=0.7):
    """启动简单Web界面 (使用 Gradio)"""
    try:
        import gradio as gr
    except ImportError:
        logger.error("请先安装 gradio: pip install gradio")
        return
    
    def chat(user_msg, history):
        if not user_msg.strip():
            return "", history
        
        history_text = [h[0] for h in history] + [h[1] for h in history]
        response = generate_response(
            model, tokenizer, user_msg,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            history=history_text,
        )
        history.append((user_msg, response))
        return "", history
    
    with gr.Blocks(title="Local Chatbot") as demo:
        gr.Markdown("# 🤖 Local Chatbot - 本地对话模型")
        gr.Markdown("基于 LoRA/QLoRA 微调的本地对话模型")
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label="输入消息", placeholder="输入你的问题...", lines=2)
        with gr.Row():
            submit_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清除")
        
        submit_btn.click(chat, [msg, chatbot], [msg, chatbot])
        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


def main():
    args = parse_args()
    
    # 加载模型
    model, tokenizer = load_model_for_inference(args)
    
    # 单轮对话
    if args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n👤 输入: {args.prompt}")
        print(f"\n🤖 回复: {response}")
        return
    
    # Web 界面
    if args.web:
        start_web_ui(
            model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        return
    
    # 交互模式
    chat_loop(
        model, tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
