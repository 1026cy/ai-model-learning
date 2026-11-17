"""
模块 1：数据张量化、分词与 Embedding 基石
对应课程：文本 → Token IDs → Dataset/DataLoader → Token/Position Embedding
前置要求：先完成 00_pytorch_tensor.py，掌握 Tensor 创建、dtype、形状与切片。

运行：python 01_data_pipeline.py

本模块只解决一个问题：模型不能直接读文字，怎样把文字变成它能训练的张量？
完成本模块后，再进入注意力机制，不要提前担心 Transformer 的其他部分。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


TEXT = (
    "To be, or not to be, that is the question.\n"
    "Whether 'tis nobler in the mind to suffer.\n"
    "语言模型根据前面的字符，学习预测下一个字符。\n"
) * 4


class CharTokenizer:
    """字符级分词器：一个字符对应一个整数 ID。"""

    def __init__(self, text):
        # sorted(set(text)) 建立确定性的词表，保证每次运行 ID 映射一致。
        self.chars = sorted(set(text))
        # print(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        # print(self.stoi)
        # print(self.itos)
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """把字符串转换为一维 LongTensor，例如 "ab" -> tensor([id_a, id_b])。"""
        # TODO 1：遍历 text，用 self.stoi 把每个字符转换为 ID。
        # 返回 torch.tensor(..., dtype=torch.long)。
        # 如果字符不在词表中，让 KeyError 直接暴露问题，不要静默替换成其他字符。
        # raise NotImplementedError("TODO 1：实现 CharTokenizer.encode")
        for i in text:


    def decode(self, token_ids):
        """把 token ID 序列还原为字符串，token_ids 可以是列表或 Tensor。"""
        # TODO 2：遍历 token_ids，用 self.itos 找到字符并拼接。
        # int(token_id) 可以同时处理 Python int 和 0 维 Tensor。
        raise NotImplementedError("TODO 2：实现 CharTokenizer.decode")


class NextTokenDataset(Dataset):
    """
    用滑动上下文窗口构造自回归样本。

    若一段 token 是 [A,B,C,D,E]，block_size=4：
        x = [A,B,C,D]
        y = [B,C,D,E]
    模型在第 0 位看到 A，要预测 B；在第 1 位看到 A,B，要预测 C。
    """

    def __init__(self, token_ids, block_size):
        super().__init__()
        assert token_ids.ndim == 1
        assert len(token_ids) > block_size
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        # 每个样本还需要额外一个 token 作为最后一个位置的标签。
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):
        """返回 x、y，两者形状均为 [block_size]。"""
        # TODO 3：按 index:index+block_size 切输入，
        # 按 index+1:index+block_size+1 切标签。
        raise NotImplementedError("TODO 3：实现上下文窗口与错位标签")


def embed_tokens(token_ids, token_embedding, position_embedding):
    """
    把离散 Token IDs 变成带位置信息的稠密向量。

    输入：token_ids [B,T]
    输出：[B,T,d_model]

    Token Embedding 表示“这个字符是什么”，Position Embedding 表示“它在第几个位置”。
    二者相加后，模型才能区分相同字符出现在不同位置的情况。
    """
    B, T = token_ids.shape
    positions = torch.arange(T, device=token_ids.device)

    # TODO 4：计算 token_embedding(token_ids) + position_embedding(positions)。
    # position_embedding(positions) 的形状是 [T,d_model]，会广播到 batch 维。
    # 返回前检查并理解三个张量的形状。
    raise NotImplementedError("TODO 4：组合 Token 与 Position Embedding")


def main():
    torch.manual_seed(42)
    tokenizer = CharTokenizer(TEXT)
    token_ids = tokenizer.encode(TEXT)

    print(f"字符数：{len(TEXT)}，词表大小：{tokenizer.vocab_size}")
    assert tokenizer.decode(token_ids) == TEXT
    print("Tokenizer 编码/解码检查通过。")

    block_size = 16
    dataset = NextTokenDataset(token_ids, block_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y = next(iter(loader))

    print("输入 x：", x.shape)  # [4,16]
    print("标签 y：", y.shape)  # [4,16]
    assert x.shape == y.shape == (4, block_size)

    # 检查每条样本的标签确实是输入右移一位后的结果。
    sample_x, sample_y = dataset[0]
    assert torch.equal(sample_y[:-1], sample_x[1:])

    d_model = 32
    token_embedding = nn.Embedding(tokenizer.vocab_size, d_model)
    position_embedding = nn.Embedding(block_size, d_model)
    embedded = embed_tokens(x, token_embedding, position_embedding)

    print("Embedding 输出：", embedded.shape)
    assert embedded.shape == (4, block_size, d_model)
    print("模块 1 验收通过：文本 → IDs → [B,T] → [B,T,d_model]。")

    # 练习：
    # 1. 修改 block_size，观察 x/y 和 position_embedding 的形状约束。
    # 2. 打印 token_embedding.weight.shape，理解它是 [vocab_size,d_model] 的参数表。
    # 3. 思考：没有 Position Embedding 时，Attention 为什么不知道字符顺序？


if __name__ == "__main__":
    main()
