"""
模块 5：训练循环、Loss、Warmup+Cosine、采样与自回归生成
对应课程：Mini-GPT 训练与推理全闭环

前置：完成 01～04，尤其是 04_mini_gpt.py。
运行：python 05_train_and_generate.py

数据流回顾：
    x[B,T] -> MiniGPT -> logits[B,T,V]
    y[B,T] 是每个位置正确的下一 token
    logits 与 y 计算交叉熵，反向传播并更新参数。
"""

import math
from importlib import import_module
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


data_lesson = import_module("01_data_pipeline")
model_lesson = import_module("04_mini_gpt")
CharTokenizer = data_lesson.CharTokenizer
NextTokenDataset = data_lesson.NextTokenDataset
MiniGPT = model_lesson.MiniGPT


def warmup_cosine_multiplier(step, warmup_steps, total_steps, min_ratio=0.1):
    """返回当前学习率相对 base_lr 的倍率，配合 LambdaLR 使用。"""
    # TODO 1：实现 warmup + cosine annealing。
    # warmup 阶段：倍率从接近 0 线性升到 1，可用 (step+1)/warmup_steps。
    # 之后：progress 从 0 到 1，余弦倍率从 1 降到 min_ratio：
    # min_ratio + (1-min_ratio) * 0.5 * (1 + cos(pi*progress))
    # 前段避免一开始大步破坏随机参数，后段逐步减小步长便于收敛。
    raise NotImplementedError("TODO 1：实现 Warmup 与 Cosine 学习率调度")


def top_k_filter(logits, k):
    """保留每行分数最高的 k 个 token，其余设为 -inf。logits: [B,V]。"""
    if k <= 0 or k >= logits.size(-1):
        return logits.clone()

    # TODO 2：torch.topk 找到第 k 大阈值，把低于阈值的 logits 填为 -inf。
    raise NotImplementedError("TODO 2：实现 Top-k 截断")


def top_p_filter(logits, p):
    """保留累计概率达到 p 所需的最小高概率候选集合。"""
    if p <= 0 or p >= 1:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # TODO 3：构造 top-p mask，并映射回原始词表顺序。
    # 先令 sorted_remove = cumulative > p；再整体右移一位，保留第一个越过 p 的 token。
    # 最后用 scatter 按 sorted_indices 映射回原位置，masked_fill 为 -inf。
    raise NotImplementedError("TODO 3：实现 Top-p 截断")


def sample_next_token(logits, temperature=0.8, top_k=0, top_p=1.0, greedy=False):
    """从最后一个位置的 logits [B,V] 中选择一个 token，返回 [B,1]。"""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature <= 0:
        raise ValueError("非贪婪采样时 temperature 必须大于 0")

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=80, temperature=0.8, top_k=20, top_p=0.9, greedy=False):
    """
    自回归生成：每次只预测一个新 token，拼接后作为下一轮上下文。
    prompt_ids: [1,T_prompt]，返回包含 prompt 的完整 ID 序列。
    """
    model.eval()
    idx = prompt_ids

    for _ in range(max_new_tokens):
        # TODO 4：完成自回归循环。
        # 1. idx_cond 只保留最后 model.block_size 个 token，避免位置 Embedding 越界。
        # 2. logits = model(idx_cond)，取最后一个位置 logits[:, -1, :]。
        # 3. 调用 sample_next_token 得到 next_id [1,1]。
        # 4. torch.cat([idx,next_id], dim=1) 拼接，进入下一轮。
        # 注意：这里生成的是 token ID，不是每个位置都重新生成整段文本。
        raise NotImplementedError("TODO 4：实现自回归生成")

    return idx


def train(model, loader, device, max_steps=300, warmup_steps=20, base_lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_multiplier(step, warmup_steps, max_steps),
    )
    data_iter = iter(loader)
    losses = []
    model.train()

    for step in range(max_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        logits = model(x)  # [B,T,V]

        # TODO 5：计算所有 batch、所有位置的平均交叉熵。
        # F.cross_entropy 需要 [N,V] 与 [N]，因此把 logits/y 展平。
        raise NotImplementedError("TODO 5：实现序列交叉熵 Loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪限制全局梯度范数，降低偶发大梯度导致训练发散的概率。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if step % 50 == 0 or step == max_steps - 1:
            print(
                f"step={step:03d}, loss={loss.item():.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    return losses


def plot_losses(losses, output="loss_curve.png"):
    """绘制训练 Loss；没有 matplotlib 时只提示，不影响模型训练结果。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过 Loss 曲线绘制。")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Mini-GPT Training Loss")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Loss 曲线已保存到 {Path(output).resolve()}")


def generate_text(model, tokenizer, prompt, device, **kwargs):
    prompt_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    output = generate(model, prompt_ids, **kwargs)
    return tokenizer.decode(output[0])


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = data_lesson.TEXT * 20
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    block_size = 32
    dataset = NextTokenDataset(token_ids, block_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
    ).to(device)

    print(f"设备：{device}，参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("\n训练前的随机输出：")
    print(generate_text(model, tokenizer, "To be", device, max_new_tokens=40))

    losses = train(model, loader, device)
    plot_losses(losses)

    print("\n训练后的采样输出：")
    print(generate_text(model, tokenizer, "To be", device, max_new_tokens=80))
    print("\n训练后的 Greedy 输出：")
    print(generate_text(model, tokenizer, "To be", device, max_new_tokens=80, greedy=True))

    # 模块 5 验收：
    # [ ] Loss 总体下降，且曲线已绘制。
    # [ ] 生成时上下文不超过 block_size。
    # [ ] 能解释 greedy、temperature、top-k、top-p 对输出的影响。
    # [ ] 能独立说明 forward、loss、backward、optimizer.step 的顺序。


if __name__ == "__main__":
    main()
