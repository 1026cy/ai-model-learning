"""
第 6 课：不依赖 PEFT，亲手理解 LoRA
技术栈：Python + PyTorch

运行方式：
    python 06_lora_from_scratch.py

LoRA（Low-Rank Adaptation）的核心不是“把大模型变小”，而是：
    冻结原始线性层 W，只学习一个低秩增量 ΔW = B @ A。

若原始权重 W 的形状为 [out_features, in_features]，秩为 r：
    A: [r, in_features]
    B: [out_features, r]

当 r 远小于输入、输出维度时，A 和 B 的参数量远小于完整 W。
本课用一个很小的线性层验证：冻结、低秩更新、合并权重三件事。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8):
        super().__init__()
        assert rank > 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 这是已经预训练好的原始层。实验中会冻结它，只训练后面的 A、B。
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)

        # A 通常随机初始化，提供低秩变换的方向。
        self.A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

        # B 初始化为 0，因此训练刚开始时 ΔW = B@A = 0。
        # 这样加入 LoRA 不会突然破坏原模型已有的输出。
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        """
        普通线性层：y = x @ W^T
        LoRA 线性层：y = x @ W^T + x @ (B@A)^T * scaling

        实际计算时不需要显式构造完整 ΔW：
            x @ A^T 得到 [..., r]
            再 @ B^T 得到 [..., out_features]
        这样更节省计算和内存。
        """
        base_output = F.linear(x, self.base.weight)

        # TODO 1：计算 LoRA 分支 lora_output = (x @ A.T @ B.T) * scaling。
        # 最后返回 base_output + lora_output。
        raise NotImplementedError("TODO 1：请实现 LoRA 前向传播")

    def merged_weight(self):
        """返回合并后的完整权重 W_merged = W + (B@A)*scaling。"""
        # TODO 2：计算合并权重。
        # base.weight: [out,in]
        # B@A:        [out,in]
        # 合并训练结果时使用 detach()，因为这里不再需要建立训练计算图。
        raise NotImplementedError("TODO 2：请实现合并权重")

    @torch.no_grad()
    def merge(self):
        """创建一个包含 LoRA 增量、可以独立推理的普通 Linear。"""
        merged = nn.Linear(self.in_features, self.out_features, bias=False)
        merged.weight.copy_(self.merged_weight())
        return merged


def count_parameters(model):
    """分别统计总参数量和可训练参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    torch.manual_seed(42)
    in_features, out_features = 64, 32
    model = LoRALinear(in_features, out_features, rank=4, alpha=8)

    total, trainable = count_parameters(model)
    base_params = in_features * out_features
    full_update_params = base_params

    print(f"原始权重参数量：{base_params:,}")
    print(f"包含 LoRA 后的总参数量：{total:,}")
    print(f"实际可训练参数量：{trainable:,}")
    print(f"若全量训练，需要训练的参数量：{full_update_params:,}")

    assert not model.base.weight.requires_grad
    assert model.A.requires_grad and model.B.requires_grad

    x = torch.randn(3, in_features)

    # B 初始化为 0，所以此时 LoRA 输出必须与原始线性层完全一致。
    initial_output = model(x)
    base_output = model.base(x)
    assert torch.allclose(initial_output, base_output, atol=1e-6)

    # 为了测试合并逻辑，手动给 B 一个非零值。
    # 真正训练时，B 会通过反向传播逐渐变成非零。
    with torch.no_grad():
        model.B.normal_(mean=0.0, std=0.02)

    # 合并前后的输出应该相同：合并只是改变存储方式，不改变模型函数。
    output_before_merge = model(x)
    merged_model = model.merge()
    output_after_merge = merged_model(x)

    max_difference = (output_before_merge - output_after_merge).abs().max().item()
    print(f"合并前后最大输出误差：{max_difference:.8f}")
    assert torch.allclose(output_before_merge, output_after_merge, atol=1e-5)

    print("LoRA 冻结、初始化和权重合并检查通过。")

    # 附加实验：
    # 1. 把 rank 改为 1、8、16，可训练参数量如何变化？
    # 2. rank 最大增加到什么值时，LoRA 参数量可能不再有优势？
    # 3. 如果 A、B 都随机初始化，训练开始时会发生什么？
    # 4. 为什么 LoRA 合并后推理不需要保留 A、B 两个矩阵？


if __name__ == "__main__":
    main()
