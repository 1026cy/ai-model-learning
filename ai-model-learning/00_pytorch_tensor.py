"""
预备模块：PyTorch Tensor 最小必备基础
目的：补齐进入 01_data_pipeline.py 之前会用到的 Tensor 操作。

运行：python 00_pytorch_tensor.py

先澄清名称：
    torch.Tensor  是张量类（大写 T）。
    torch.tensor  是创建张量的函数（小写 t），返回一个 Tensor 实例。

本模块只讲后面马上会用的操作，不扩展到自动求导或神经网络。
"""

import torch


def create_token_ids():
    """创建一维 token ID 张量，这是后续文本数值化的起点。"""
    # TODO 1：用 torch.tensor 创建包含 [5, 12, 7, 0, 9, 3] 的张量。
    # 必须指定 dtype=torch.long，因为 Embedding 的索引需要 64 位整数。
    # 结构：return torch.tensor(数据列表, dtype=...)
    # raise NotImplementedError("TODO 1：创建 LongTensor")
    return torch.tensor([5, 12, 7, 0, 9, 3], dtype=torch.long)

def inspect_tensor(x):
    """
    读取张量最常用的四个属性：
        shape：每个维度的大小
        ndim：维度数量
        dtype：元素类型
        numel()：元素总数（注意这是方法，要加括号）
    """
    # TODO 2：按顺序 return x.shape, x.ndim, x.dtype, x.numel()
    # raise NotImplementedError("TODO 2：读取 Tensor 基本信息")
    # print("x.shape =", x.shape, "x.ndim =", x.ndim, "x.dtype =", x.dtype, "x.numel() =", x.numel())
    return x.shape, x.ndim, x.dtype, x.numel()

def index_and_slice(x):
    """
    练习后续滑动窗口会用的索引和切片。
    x 为 [5,12,7,0,9,3] 时：
        first = 5
        window = [12,7,0]
        last_two = [9,3]
    """
    # TODO 3：分别用 x[0]、x[1:4]、x[-2:] 完成。
    # Python/PyTorch 切片包含起点、不包含终点，所以 x[1:4] 取索引 1、2、3。
    # 'raise NotImplementedError("TODO 3：索引与切片")
    first = x[0]
    window = x[1:4]
    last_two = x[-2:]
    return first, window, last_two


def reshape_tensor(x):
    """把 6 个元素的一维张量变成 2 行 3 列，再恢复成一维。"""
    # TODO 4：matrix = x.reshape(2,3)，flat = matrix.reshape(-1)。
    # -1 表示让 PyTorch 根据元素总数自动推导这一维，返回 matrix, flat。
    raise NotImplementedError("TODO 4：改变张量形状")


def add_batch_dimension(x):
    """
    把 [6] 变成 [1,6]，模拟给单条 token 序列增加 batch 维。
    模型通常期待输入 [Batch,Sequence]，而不是只有 [Sequence]。
    """
    # TODO 5：return x.unsqueeze(0)，表示在第 0 维插入一个大小为 1 的维度。
    raise NotImplementedError("TODO 5：增加 batch 维")


def broadcasting_demo(x):
    """观察广播：标量可以作用到张量的每个元素。"""
    result = x + 10
    print("x + 10 =", result.tolist())
    print("x 的原始值 =", x.tolist(), "（加法没有原地修改 x）")


def next_token_preview(x):
    """只做形状预览，不要求现在掌握语言模型。"""
    inputs = x[:-1]   # 去掉最后一个：本例为 [5,12,7,0,9]
    targets = x[1:]   # 去掉第一个：本例为 [12,7,0,9,3]
    print("输入预览：", inputs.tolist())
    print("错位标签：", targets.tolist())


def main():
    token_ids = create_token_ids()
    shape, ndim, dtype, count = inspect_tensor(token_ids)

    print("token_ids =", token_ids)
    print("shape =", shape, "ndim =", ndim, "dtype =", dtype, "numel =", count)
    assert shape == (6,)
    assert ndim == 1
    assert dtype == torch.long
    assert count == 6

    first, window, last_two = index_and_slice(token_ids)
    print("first =", first, "window =", window, "last_two =", last_two)
    assert first.item() == 5
    assert window.tolist() == [12, 7, 0]
    assert last_two.tolist() == [9, 3]

    matrix, flat = reshape_tensor(token_ids)
    print("2x3 matrix：\n", matrix)
    assert matrix.shape == (2, 3)
    assert flat.shape == (6,)
    assert torch.equal(flat, token_ids)

    batched = add_batch_dimension(token_ids)
    print("增加 batch 维后的形状：", batched.shape)
    assert batched.shape == (1, 6)

    broadcasting_demo(token_ids)
    next_token_preview(token_ids)
    print("预备模块验收通过，可以进入 01_data_pipeline.py。")

    # 自测：不用运行代码，先回答再验证。
    # 1. torch.tensor([1,2,3]).dtype 和指定 torch.long 有何关系？
    # 2. x[1:4] 为什么不是取到索引 4？
    # 3. [6]、[1,6]、[2,3] 的元素总数分别是多少？
    # 4. numel 和 numel() 哪个才是正确调用？


if __name__ == "__main__":
    main()
