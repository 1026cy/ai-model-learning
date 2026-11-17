"""
模块 4：堆叠 Decoder Block 并组装 Mini-GPT
对应课程：Token/Position Embedding → N 个 Block → Final LN → LM Head

前置：完成 03_transformer_block.py。
运行：python 04_mini_gpt.py

本模块只搭建模型骨架并验证 forward，训练循环和 generate 留到模块 5。
"""

from importlib import import_module
import torch
import torch.nn as nn


attention_lesson = import_module("02_attention")
block_lesson = import_module("03_transformer_block")
create_causal_mask = attention_lesson.create_causal_mask
TransformerBlock = block_lesson.TransformerBlock


class MiniGPT(nn.Module):
    """纯 Decoder 自回归模型：每个位置只根据当前及过去 token 预测下一 token。"""

    def __init__(
        self,
        vocab_size,
        block_size=64,
        d_model=128,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        # TODO 1：用 nn.ModuleList 堆叠 num_layers 个 TransformerBlock。
        # 不要用普通 Python list，否则子模块参数无法被 model.parameters() 正确收集。
        raise NotImplementedError("TODO 1：堆叠 Decoder Blocks")

        self.ln_final = nn.LayerNorm(d_model)

        # LM Head 把每个位置的隐藏向量映射为整个词表的 logits。
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 对所有子模块应用统一初始化。
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """小标准差初始化避免初始激活/梯度过大，帮助训练初期稳定收敛。"""
        # TODO 2：对 nn.Linear 和 nn.Embedding 的 weight 使用 std=0.02 的正态初始化。
        # 若 Linear 有 bias，则将 bias 置 0。LayerNorm 保持 PyTorch 默认即可。
        raise NotImplementedError("TODO 2：实现模型参数初始化")

    def forward(self, idx):
        """
        idx: [B,T] 的 token IDs。
        返回：logits [B,T,vocab_size]。这里不计算 loss，保持模型职责单一。
        """
        B, T = idx.shape
        assert T <= self.block_size, "上下文长度超过 block_size"
        positions = torch.arange(T, device=idx.device)

        # TODO 3：完成完整前向传播：
        # token embedding + position embedding
        # -> 依次经过 self.blocks（传入同一个 causal mask）
        # -> self.ln_final
        # -> self.lm_head
        # 返回 logits。
        raise NotImplementedError("TODO 3：实现 MiniGPT.forward")


def main():
    torch.manual_seed(42)
    vocab_size, B, T = 80, 2, 16
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )

    idx = torch.randint(0, vocab_size, (B, T))
    logits = model(idx)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"模型参数量：{param_count:,}")
    print("输入：", idx.shape)
    print("Logits：", logits.shape)
    assert logits.shape == (B, T, vocab_size)
    assert torch.isfinite(logits).all()
    assert len(model.blocks) == 2

    # 再验证一次反向传播，确保整个模型骨架可用于后续训练。
    logits.square().mean().backward()
    assert model.token_embedding.weight.grad is not None
    print("模块 4 验收通过：Mini-GPT 骨架可前向、可反向、输出形状正确。")

    # 练习：
    # 1. 把 num_layers 改为 4，观察参数量和显存变化。
    # 2. 为什么 LM Head 输出维度必须是 vocab_size？
    # 3. 尝试把初始化 std 改为 0.5，观察初始 logits 是否出现数值异常。


if __name__ == "__main__":
    main()
