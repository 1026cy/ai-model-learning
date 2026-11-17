"""
模块 2：缩放点积注意力与多头注意力
对应课程：Q/K/V → 缩放点积 → Causal Mask → Softmax → 多头拆分与合并

运行：python 02_attention.py

阅读顺序：先看 scaled_dot_product_attention，再看 MultiHeadAttention。
本模块不涉及 Transformer Block，先把注意力这一个计算单元彻底写清楚。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_causal_mask(T, device=None):
    """
    创建 [T,T] 下三角布尔矩阵，True 表示允许关注。
    第 t 行只允许看到 0～t，不能看到 t+1 之后的未来 token。
    """
    # TODO 1：用 torch.tril(torch.ones(..., dtype=torch.bool)) 创建下三角掩码。
    raise NotImplementedError("TODO 1：创建 Causal Mask")


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    输入：q/k/v [B,H,T,D]
    输出：context [B,H,T,D]，weights [B,H,T,T]

    Q 与 K 决定“应该关注谁”，V 提供“真正取出来混合的信息”。
    缩放 1/sqrt(D) 防止 D 较大时点积数值过大，导致 softmax 饱和、梯度过小。
    """
    D = q.size(-1)

    # TODO 2：scores = q @ k.transpose(-2,-1) / sqrt(D)。
    # scores[b,h,i,j] 表示第 i 个 Query 对第 j 个 Key 的相关性分数。
    raise NotImplementedError("TODO 2：计算缩放点积分数")

    if mask is not None:
        # 先填极小值再 softmax，未来位置的权重才会接近 0。
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # TODO 3：weights = softmax(scores, dim=-1)，context = weights @ v。
    # 最后一个维度是 Key 位置；每个 Query 的权重总和为 1。
    raise NotImplementedError("TODO 3：计算注意力权重与 Value 加权和")

    return context, weights


class MultiHeadAttention(nn.Module):
    """把 d_model 拆成多个头并行计算，再拼接回原始维度。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 同一个输入分别投影成 Q/K/V，让模型自己学习三种表示。
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """[B,T,C] -> [B,H,T,D]，其中 C=H*D。"""
        B, T, C = x.shape

        # TODO 4：x.view(B,T,self.num_heads,self.head_dim).transpose(1,2)。
        # view 负责拆通道，transpose 负责把“头”放到序列维度之前。
        raise NotImplementedError("TODO 4：拆分多头")

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        context, weights = scaled_dot_product_attention(q, k, v, mask)

        # TODO 5：[B,H,T,D] -> [B,T,H,D] -> [B,T,C]。
        # transpose 后调用 contiguous()，再 view(B,T,C)，最后经过 out_proj。
        # 返回 (输出, 注意力权重)，权重保留用于检查 mask。
        raise NotImplementedError("TODO 5：合并多头并输出")


def main():
    torch.manual_seed(42)
    B, T, d_model, H = 2, 6, 32, 4
    x = torch.randn(B, T, d_model)
    mask = create_causal_mask(T, x.device)

    attention = MultiHeadAttention(d_model, H)
    output, weights = attention(x, mask)

    print("输入：", x.shape)
    print("输出：", output.shape)
    print("权重：", weights.shape)
    assert output.shape == (B, T, d_model)
    assert weights.shape == (B, H, T, T)

    # 每行权重总和为 1；上三角未来位置必须全为 0。
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(weights.masked_select(future) == 0)
    print("模块 2 验收通过：形状、归一化和因果掩码均正确。")

    # 练习：
    # 1. 将 H 改为 1/2/8，解释 head_dim 和 weights.shape 如何变化。
    # 2. 暂时去掉缩放，比较 scores 的绝对值和 softmax 分布。
    # 3. 去掉 mask 后生成文本为什么等于“偷看答案”？


if __name__ == "__main__":
    main()
