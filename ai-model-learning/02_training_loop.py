"""
第 2 课：亲手写一个完整训练循环
技术栈：Python + PyTorch（只用 Tensor 和 autograd）

运行方式：
    python 02_training_loop.py

任务：
用随机初始化的 w、b 拟合 y = 3x + 2。
本课故意不使用 nn.Linear、torch.optim 或 Trainer，因为它们是训练循环的封装，
而不是训练本身。只有亲手写过一次，才知道框架替你做了什么。
"""

import torch


def make_data(num_samples=200, noise_std=0.05):
    """生成带少量噪声的线性数据，使任务更接近真实训练。"""
    torch.manual_seed(42)  # 固定随机种子，方便复现实验和比较结果

    # unsqueeze(1) 把形状从 [N] 变为 [N, 1]。
    # 对线性层而言，通常最后一维表示特征数，这里每个样本只有一个特征。
    x = torch.linspace(-1.0, 1.0, num_samples).unsqueeze(1)
    noise = torch.randn(num_samples, 1) * noise_std
    y = 3.0 * x + 2.0 + noise
    return x, y


def linear_forward(x, w, b):
    """
    线性模型的前向传播。

    x: [batch_size, 1]
    w: [1, 1]，表示一个输入特征到一个输出特征的权重
    b: [1]，每个输出特征一个偏置
    返回: [batch_size, 1]
    """
    # TODO 1：实现 x @ w + b。
    # @ 是矩阵乘法；b 会通过广播作用到 batch 中的每个样本。
    raise NotImplementedError("TODO 1：请实现 linear_forward")


def mse_loss(y_pred, y_true):
    """返回所有样本、所有输出维度上的平均平方误差。"""
    # TODO 2：再次亲手实现 MSE，加深印象。
    raise NotImplementedError("TODO 2：请实现 mse_loss")


def train(x, y, learning_rate=0.1, num_steps=300):
    """使用全量数据做梯度下降，并手动更新参数。"""
    # nn.Parameter 只是一个方便标记“这是可训练参数”的 Tensor 包装。
    # 初始值故意离真实答案 w=3、b=2 较远。
    w = torch.nn.Parameter(torch.tensor([[0.0]]))
    b = torch.nn.Parameter(torch.tensor([0.0]))

    for step in range(num_steps):
        # ---------- 1. 前向传播 ----------
        y_pred = linear_forward(x, w, b)
        loss = mse_loss(y_pred, y)

        # ---------- 2. 清空旧梯度 ----------
        # PyTorch 默认会累加梯度，而不是自动覆盖。
        # 如果不清空，上一轮的梯度会混入这一轮。
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()

        # ---------- 3. 反向传播 ----------
        loss.backward()

        # ---------- 4. 手动更新参数 ----------
        # 梯度下降公式：parameter = parameter - learning_rate * gradient
        # 更新本身不需要建立计算图，所以放在 no_grad 中。
        with torch.no_grad():
            # TODO 3：分别更新 w 和 b。
            # 注意使用原位操作 w -= ...、b -= ...，不要重新创建普通 Tensor。
            raise NotImplementedError("TODO 3：请实现参数更新")

        if step % 30 == 0 or step == num_steps - 1:
            print(
                f"step={step:03d} loss={loss.item():.6f} "
                f"w={w.item():.4f} b={b.item():.4f}"
            )

    return w.detach(), b.detach(), loss.item()


def main():
    x, y = make_data()
    w, b, final_loss = train(x, y)

    print("\n训练完成：")
    print("学习到的 w =", w.item(), "（真实值约为 3）")
    print("学习到的 b =", b.item(), "（真实值约为 2）")
    print("最终 loss =", final_loss)

    # 数据包含噪声，因此不要求完全等于 3 和 2。
    assert abs(w.item() - 3.0) < 0.2
    assert abs(b.item() - 2.0) < 0.2

    # 附加实验：每次只修改一个变量，观察并记录结果。
    # 1. learning_rate 改为 1.5，会发生什么？为什么？
    # 2. learning_rate 改为 0.0001，相同步数内为什么学不好？
    # 3. 删除 w.grad.zero_() 和 b.grad.zero_()，结果会怎样？
    # 4. 把噪声 noise_std 改成 0，最终参数是否更接近精确答案？


if __name__ == "__main__":
    main()
