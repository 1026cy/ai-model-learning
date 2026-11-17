"""
模块 3：Pre-LN、残差连接、前馈网络与 TransformerBlock
对应课程：Attention 子层 + FFN 子层 → 完整 Block

前置：完成 02_attention.py。
运行：python 03_transformer_block.py

本模块的关键不是新公式，而是信息如何流动：
Pre-LN 让归一化发生在子层之前，残差连接保留原始信息和梯度通路。
"""

from importlib import import_module
import torch
import torch.nn as nn


attention_lesson = import_module("02_attention")
MultiHeadAttention = attention_lesson.MultiHeadAttention
create_causal_mask = attention_lesson.create_causal_mask


class FeedForward(nn.Module):
    """
    对每个 token 位置独立做非线性变换。
    Attention 负责 token 之间的信息交换，FFN 负责每个位置内部的特征组合。
    """

    def __init__(self, d_model, expansion=4):
        super().__init__()
        # TODO 1：用 nn.Sequential 构建：
        # Linear(d_model, expansion*d_model) -> GELU -> Linear(expansion*d_model, d_model)。
        # 升维扩大特征组合空间，最后必须降回 d_model 才能做残差相加。
        raise NotImplementedError("TODO 1：实现 FFN 升维、激活与降维")

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, mask=None):
        """输入/输出都必须是 [B,T,d_model]，残差分支不能改变形状。"""
        # TODO 2：实现 Pre-LN Attention 残差：
        # attn_out, _ = self.attn(self.ln_attn(x), mask)
        # x = x + attn_out
        raise NotImplementedError("TODO 2：实现 Attention 的 Pre-LN 与残差")

        # TODO 3：实现 Pre-LN FFN 残差：x = x + self.ffn(self.ln_ffn(x))。
        raise NotImplementedError("TODO 3：实现 FFN 的 Pre-LN 与残差")

        return x


def main():
    torch.manual_seed(42)
    B, T, d_model, H = 2, 8, 32, 4
    x = torch.randn(B, T, d_model, requires_grad=True)
    mask = create_causal_mask(T)

    block = TransformerBlock(d_model, H)
    output = block(x, mask)

    print("Block 输入：", x.shape)
    print("Block 输出：", output.shape)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    # 做一次反向传播，验证残差路径和子层参数都能收到梯度。
    output.square().mean().backward()
    assert x.grad is not None
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"{name} 没有收到梯度"

    print("模块 3 验收通过：Block 形状正确，前向稳定，反向梯度连通。")

    # 练习：
    # 1. 把 Pre-LN 改成 Post-LN，画出两者数据流差异。
    # 2. 删除残差连接后堆叠更多 Block，比较梯度与训练稳定性。
    # 3. 为什么 FFN 的最后一层必须输出 d_model 维？


if __name__ == "__main__":
    main()
