"""
第 5 课：用 Attention 拼出一个 mini GPT，并从随机权重开始训练
技术栈：Python + PyTorch + 第 4 课实现的 MultiHeadSelfAttention

前置要求：
先完成 04_self_attention.py。本文件会复用其中的注意力模块。

运行方式：
    python 05_mini_gpt.py

本课包含一个 GPT 的基本组成：
    token embedding + position embedding
    -> N 个 causal Transformer block
    -> LayerNorm
    -> language model head
    -> next-token loss

模型很小，数据也很少，目标不是得到实用聊天机器人，而是验证整条训练链路。
"""

from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F


# 文件名以数字开头，不能写成 from 04_self_attention import ...。
# import_module 可以按文件名加载同一目录下的第 4 课代码。
attention_lesson = import_module("04_self_attention")
MultiHeadSelfAttention = attention_lesson.MultiHeadSelfAttention
create_causal_mask = attention_lesson.create_causal_mask


TEXT = (
    "春天来了，风吹过树林。小鸟在枝头唱歌，溪水向东流。\n"
    "秋天到了，树叶慢慢变黄。晚风吹过田野，月亮升上天空。\n"
) * 8  # 重复短文本，方便小模型先学会局部规律并出现过拟合


class FeedForward(nn.Module):
    """每个 token 位置独立执行的前馈网络，不负责 token 之间的信息交换。"""

    def __init__(self, d_model, hidden_multiplier=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_multiplier * d_model),
            nn.GELU(),
            nn.Linear(hidden_multiplier * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block：先归一化，再做 Attention/FFN，并保留残差连接。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, mask):
        # TODO 1：完成 Attention 子层和残差连接。
        # self.attn 返回 (attn_output, attention_weights)，这里只需要 attn_output。
        # 公式：x = x + Attention(LayerNorm(x))
        raise NotImplementedError("TODO 1：请实现 Attention 残差子层")

        # TODO 2：完成前馈子层和残差连接。
        # 公式：x = x + FFN(LayerNorm(x))
        raise NotImplementedError("TODO 2：请实现 FFN 残差子层")

        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=64, d_model=128, num_heads=4, num_layers=2):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        # token embedding：把离散字符 ID 映射为连续向量。
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 本练习使用可学习的位置 embedding，让模型知道 token 的先后顺序。
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)

        # 输出每个位置对应词表中所有字符的 logits，用于预测下一个字符。
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        idx: [B,T]，每个位置是当前 token ID。
        targets: [B,T]，每个位置是正确的下一个 token ID。
        """
        B, T = idx.shape
        assert T <= self.block_size, "输入序列不能超过模型支持的最大上下文长度"

        positions = torch.arange(T, device=idx.device)  # [T]

        # TODO 3：把 token embedding 和 position embedding 相加。
        # token_embedding(idx) 为 [B,T,C]；position_embedding(positions) 为 [T,C]。
        # 给位置向量增加 batch 维度后，广播相加即可。
        raise NotImplementedError("TODO 3：请组合 token 与位置表示")

        mask = create_causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # [B,T,vocab_size]
        loss = None

        if targets is not None:
            # TODO 4：计算整个 batch、整个序列上的 next-token 交叉熵。
            # cross_entropy 默认要求预测形状为 [N,C]，目标形状为 [N]。
            # 因此需要把 [B,T,V] 展平为 [B*T,V]，把 targets 展平为 [B*T]。
            raise NotImplementedError("TODO 4：请计算序列语言模型 loss")

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.8):
        """自回归生成：每次只根据已有序列预测并追加一个新 token。"""
        self.eval()

        for _ in range(max_new_tokens):
            # TODO 5：裁剪上下文，使长度不超过 self.block_size。
            # 提示：只保留 idx 最后 block_size 个 token。
            raise NotImplementedError("TODO 5：请实现生成时的上下文裁剪")

            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / temperature  # 只取最后一个位置
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            idx = torch.cat([idx, next_id], dim=1)

        return idx


def get_batch(data, block_size, batch_size, device):
    """
    从一长串 token 中随机截取训练片段。
    对起点 i：输入为 data[i:i+T]，目标为 data[i+1:i+T+1]。
    """
    # TODO 6：随机选择 batch_size 个起点，并堆叠 x、y。
    # 起点范围必须保证 i + block_size + 1 <= len(data)。
    # x、y 都是 [batch_size, block_size]，最后移到 device。
    raise NotImplementedError("TODO 6：请实现随机序列批次")


def main():
    torch.manual_seed(42)
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(
        vocab_size=len(chars),
        block_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
    ).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"设备：{device}")
    print(f"词表大小：{len(chars)}")
    print(f"模型参数量：{parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    for step in range(500):
        xb, yb = get_batch(data, block_size=64, batch_size=16, device=device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0 or step == 499:
            print(f"step={step:03d}, loss={loss.item():.4f}")

    start = torch.tensor([[stoi["春"]]], dtype=torch.long, device=device)
    output = model.generate(start, max_new_tokens=80)
    print("\n生成结果：")
    print("".join(itos[int(i)] for i in output[0]))

    # 附加实验：
    # 1. 把 num_layers 改为 0，观察只有 embedding + linear 时的效果。
    # 2. 删除位置 embedding，模型可能丢失什么能力？
    # 3. 把 temperature 改为 0.2 和 1.5，比较生成结果的多样性。
    # 4. 增加 TEXT 的多样性，观察“复述训练文本”和“组合新句子”的区别。


if __name__ == "__main__":
    main()
