"""
第 1 课：Tensor、自动求导与梯度
技术栈：Python + PyTorch（不需要 GPU，也不需要其他第三方库）

运行方式：
    python 01_pytorch_autograd.py

本课目标：
1. 理解模型输入、权重和中间结果都可以表示为 Tensor。
2. 理解 loss.backward() 做了什么。
3. 能手写均方误差，并用链式法则手动检查 PyTorch 算出的梯度。

学习建议：
先阅读全部注释，再完成两个 TODO。不要直接搜索答案；可以先在纸上推导。
"""

import torch


def tensor_shape_demo():
    """观察三维张量的形状。语言模型中经常出现 [batch, sequence, feature]。"""
    # arange 生成 0～23，reshape 后形状为 [2, 3, 4]。
    # 可以暂时把它理解为：2 个样本，每个样本 3 个 token，每个 token 4 个特征。
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    print("x.shape =", x.shape)
    print("x[0].shape =", x[0].shape)       # 第一个样本：[3, 4]
    print("x[0, 0] =", x[0, 0])             # 第一个样本的第一个 token
    print("x.device =", x.device)           # 默认在 CPU 上
    print("x.dtype =", x.dtype)             # 默认使用 float32


def mse_loss(y_pred, y_true):
    """
    计算均方误差 Mean Squared Error。

    数学定义：
        loss = mean((y_pred - y_true)^2)

    参数：
        y_pred: 模型预测值，任意形状的浮点 Tensor
        y_true: 真实目标值，形状与 y_pred 相同

    返回：
        一个标量 Tensor（shape 为 []），但保留计算图，能够执行 backward()
    """
    # TODO 1：用 PyTorch 运算实现上面的公式。
    # 提示：先计算差值，再平方，最后调用 mean()。
    # 不要使用 torch.nn.functional.mse_loss，本练习的目的就是亲手实现。
    raise NotImplementedError("TODO 1：请实现 mse_loss")


def manual_linear_gradients(x, y_true, w, b):
    """
    不使用 autograd，手动计算线性模型 y_pred = w*x + b 的梯度。

    这里的 x、w、b 都是一维或标量 Tensor，loss 使用所有元素的 MSE。
    设 N 为元素总数，则：
        error = w*x + b - y_true
        d loss / d y_pred = 2 * error / N

    根据链式法则：
        d y_pred / d w = x
        d y_pred / d b = 1

    所以：
        grad_w = sum((2 * error / N) * x)
        grad_b = sum(2 * error / N)

    返回：(grad_w, grad_b)
    """
    # TODO 2：按照上面的推导计算 grad_w 和 grad_b。
    # 注意：这里只是做普通数值计算，不需要 backward()。
    raise NotImplementedError("TODO 2：请实现 manual_linear_gradients")


def autograd_demo():
    """对比自动求导与手动求导，两者应该非常接近。"""
    # 固定输入，避免每次结果不同。
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([5.1, 8.0, 11.2, 13.9])

    # requires_grad=True 表示：我希望 PyTorch 记录与这些参数有关的计算。
    # 模型训练时，需要学习的是 w 和 b，而不是输入 x。
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    # 前向传播：从参数和输入得到预测值，再得到 loss。
    y_pred = w * x + b
    loss = mse_loss(y_pred, y_true)

    # 反向传播：PyTorch 沿计算图应用链式法则。
    # 执行后，w.grad 和 b.grad 才有值。
    loss.backward()

    # 用你自己的公式再算一次，验证 autograd 并不是“黑箱魔法”。
    expected_w, expected_b = manual_linear_gradients(x, y_true, w.detach(), b.detach())

    print("\nloss =", loss.item())
    print("autograd: w.grad =", w.grad.item(), "b.grad =", b.grad.item())
    print("manual  : grad_w =", expected_w.item(), "grad_b =", expected_b.item())

    # allclose 用来容忍浮点计算中的微小误差。
    assert torch.allclose(w.grad, expected_w, atol=1e-6)
    assert torch.allclose(b.grad, expected_b, atol=1e-6)
    print("梯度检查通过。")


def no_grad_demo():
    """理解为什么推理和手动更新参数时经常使用 no_grad。"""
    w = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2.0)

    normal_result = w * x
    print("\n普通计算的 requires_grad =", normal_result.requires_grad)

    # no_grad 中的运算不会被记录到计算图，能减少不必要的内存开销。
    with torch.no_grad():
        inference_result = w * x
        print("no_grad 中的 requires_grad =", inference_result.requires_grad)

    # detach() 返回一个共享底层数据、但不再连接原计算图的新 Tensor。
    detached_result = (w * x).detach()
    print("detach 后的 requires_grad =", detached_result.requires_grad)


if __name__ == "__main__":
    tensor_shape_demo()
    autograd_demo()
    no_grad_demo()

    # 附加思考（不需要写代码）：
    # 1. 如果忘记调用 loss.backward()，w.grad 会是什么？
    # 2. 如果连续调用两次 backward()，为什么可能报错或不符合预期？
    # 3. 为什么验证模型时通常使用 torch.no_grad()？
