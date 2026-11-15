"""
第 4 课：亲手实现 Causal Self-Attention
技术栈：Python + PyTorch（矩阵乘法、softmax、nn.Linear）

运行方式：
    python 04_self_attention.py

本课只聚焦 Transformer 最核心的计算：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V

先完成 scaled_dot_product_attention，再完成多头张量变换。
建议使用纸笔跟踪每个 Tensor 的 shape，这是理解 Attention 最关键的练习。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_causal_mask(seq_len, device=None):
    """
    创建下三角布尔掩码，形状 [seq_len, seq_len]。

    True 表示允许关注，False 表示必须屏蔽。
    第 0 个 token 只能看到自己；第 1 个能看到 0、1；依此类推。
    这样模型在预测下一个 token 时，不能偷看未来的信息。

    例如 seq_len=4：
        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
    """
    # TODO 1：用 torch.tril 和 torch.ones 创建上述布尔矩阵。
    # dtype 使用 torch.bool，device 使用参数 device。
    raise NotImplementedError("TODO 1：请创建 causal mask")


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    计算缩放点积注意力。

    输入形状约定：
        q: [B, H, T, D]  Query，当前位置想寻找什么信息
        k: [B, H, T, D]  Key，每个位置能提供什么信息的索引
        v: [B, H, T, D]  Value，每个位置实际携带的内容
        mask: 可广播到 [B, H, T, T] 的布尔掩码，True 表示允许关注

    返回：
        context: [B, H, T, D]，聚合 Value 后的结果
        weights: [B, H, T, T]，每个 Query 对所有 Key 的注意力概率
    """
    d_k = q.size(-1)

    # TODO 2：计算注意力分数 scores = Q @ K^T / sqrt(d_k)。
    # k.transpose(-2, -1) 会把 [B,H,T,D] 变为 [B,H,D,T]。
    # 矩阵乘法后 scores 的形状应为 [B,H,T,T]。
    raise NotImplementedError("TODO 2：请计算缩放的注意力分数")

    if mask is not None:
        # 被屏蔽的位置要变成极小值，使 softmax 后概率接近 0。
        # 不能简单填 0，因为 exp(0)=1，仍然会得到注意力概率。
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # TODO 3：对最后一个维度做 softmax，得到每行总和为 1 的 weights。
    # 然后计算 context = weights @ v。
    raise NotImplementedError("TODO 3：请计算注意力权重和加权和")

    return context, weights


class MultiHeadSelfAttention(nn.Module):
    """把 d_model 拆成多个头，让每个头在不同的表示子空间中学习关系。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 用一个矩阵同时产生 Q、K、V，比三个独立矩阵更方便，也通常更高效。
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # 多头拼接后再做一次线性变换，让不同头的信息充分混合。
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        x: [B, T, C]
            B = batch size
            T = sequence length
            C = d_model = channels / features
        """
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)  # 三者都是 [B,T,C]

        # TODO 4：分别把 q、k、v 从 [B,T,C] 变为 [B,H,T,D]。
        # 推荐步骤（以 q 为例）：
        #   q.view(B, T, H, D)     先把 C 拆成 H 个头，每头 D 维
        #   .transpose(1, 2)       再把头维度放到序列维度之前
        # 对 k、v 做相同处理。
        raise NotImplementedError("TODO 4：请完成多头拆分")

        context, weights = scaled_dot_product_attention(q, k, v, mask)

        # TODO 5：把 context 从 [B,H,T,D] 还原为 [B,T,C]。
        # 步骤：transpose(1,2) -> contiguous() -> view(B,T,C)。
        # transpose 后内存不一定连续，因此 view 前通常需要 contiguous()。
        raise NotImplementedError("TODO 5：请合并多头")

        return self.out_proj(context), weights


def main():
    torch.manual_seed(42)
    B, T, C, H = 2, 5, 16, 4
    x = torch.randn(B, T, C)
    mask = create_causal_mask(T, x.device)

    attention = MultiHeadSelfAttention(d_model=C, num_heads=H)
    output, weights = attention(x, mask)

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
    print("注意力权重形状：", weights.shape)

    assert output.shape == (B, T, C)
    assert weights.shape == (B, H, T, T)

    # 每个 Query 对所有允许位置的注意力概率之和应该约为 1。
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # 上三角（不包括对角线）是未来位置，经过 mask 后权重必须为 0。
    future_positions = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(weights.masked_select(future_positions) == 0)

    print("形状、概率归一化和因果掩码检查均通过。")

    # 附加实验：
    # 1. 把 H 改成 1、2、8，权重形状分别是什么？
    # 2. 暂时去掉缩放项 sqrt(d_k)，观察权重是否更容易变得尖锐。
    # 3. 注释掉 masked_fill，模型为什么会在训练时“作弊”？


if __name__ == "__main__":
    main()
