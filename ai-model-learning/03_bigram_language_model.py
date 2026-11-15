"""
第 3 课：最小的神经语言模型——Bigram
技术栈：Python + PyTorch（nn.Embedding、交叉熵、Adam）

运行方式：
    python 03_bigram_language_model.py

Bigram 模型只根据“当前一个字符”预测“下一个字符”。
它没有 Attention，也看不到更长的上下文，因此生成效果有限，这是设计如此。
它的价值是让你看清语言模型最核心的训练目标：next-token prediction。
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用字符级建模，避免先学习复杂的 BPE tokenizer。
# 中文每个汉字在这里都被视为一个 token，换行符也算 token。
TEXT = """小明喜欢机器学习。
小红喜欢深度学习。
小明学习数学和编程。
小红学习数学和音乐。
神经网络通过数据学习参数。
语言模型学习预测下一个字符。
"""


def build_vocab(text):
    """建立字符与整数 ID 之间的双向映射。"""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to index
    itos = {i: ch for ch, i in stoi.items()}      # index to string
    return stoi, itos


def encode(text, stoi):
    """把字符串转换为 LongTensor，因为 Embedding 需要整数索引。"""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(indices, itos):
    """把 token ID 序列还原为文本。indices 可以包含 0 维 Tensor。"""
    return "".join(itos[int(i)] for i in indices)


def make_training_pairs(text, stoi):
    """
    从连续文本构造输入和目标。

    例如字符序列为：A B C D
    模型输入 x 应为：  A B C
    预测目标 y 应为：  B C D

    因此不需要人工标注答案：文本的下一个字符天然就是标签。

    返回：
        x: [token_count - 1]，每个元素是当前字符 ID
        y: [token_count - 1]，每个元素是下一个字符 ID
    """
    data = encode(text, stoi)

    # TODO 1：用 Tensor 切片构造 x 和 y。
    # 提醒：x 从第一个字符开始，y 从第二个字符开始，两者长度必须一致。
    raise NotImplementedError("TODO 1：请构造语言模型的输入与标签")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 对 bigram 来说，Embedding 表可以直接当作 logits 表使用。
        # 第 i 行表示“看到第 i 个字符后，词表中每个字符作为下一字符的分数”。
        # 输出形状：[batch_size, vocab_size]。
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: [B]，当前字符 ID。
        targets: [B]，正确的下一个字符 ID；仅训练时提供。
        """
        logits = self.token_logits(idx)
        loss = None

        if targets is not None:
            # TODO 2：计算交叉熵。
            # 提示：F.cross_entropy 的输入顺序是 (预测 logits, 正确类别 ID)。
            # logits 不需要先 softmax，因为 cross_entropy 内部会处理。
            raise NotImplementedError("TODO 2：请计算交叉熵 loss")

        return logits, loss

    @torch.no_grad()
    def generate(self, start_id, max_new_chars=60):
        """从一个字符开始，不断采样下一个字符。"""
        self.eval()
        generated = [int(start_id)]
        idx = torch.tensor([start_id], dtype=torch.long, device=self.token_logits.weight.device)

        for _ in range(max_new_chars):
            # 每次只把当前字符送入模型，得到下一字符的 logits。
            logits, _ = self(idx)

            # TODO 3：把 logits 转成概率并采样。
            # 第一步：probs = softmax(logits, dim=-1)
            # 第二步：用 torch.multinomial(probs, num_samples=1) 随机抽取一个 ID。
            # 不建议总是 argmax，因为重复选择最高概率字符很容易产生机械循环。
            raise NotImplementedError("TODO 3：请实现概率归一化与采样")

            # 采样结果的形状是 [1, 1]，转成 [1] 后作为下一轮输入。
            idx = next_id.squeeze(-1)
            generated.append(int(idx.item()))

        return generated


def get_batch(x, y, batch_size, device):
    """从所有相邻字符对中随机抽取一个 mini-batch。"""
    # randint 的上限是 len(x)，所以 i + batch_size 不会超过数据长度。
    i = torch.randint(0, len(x), (batch_size,))
    return x[i].to(device), y[i].to(device)


def main():
    torch.manual_seed(42)
    random.seed(42)

    stoi, itos = build_vocab(TEXT)
    vocab_size = len(stoi)
    x, y = make_training_pairs(TEXT, stoi)

    print(f"文本字符数：{len(TEXT)}")
    print(f"词表大小：{vocab_size}")
    print(f"训练样本数（相邻字符对）：{len(x)}")
    print("前 5 个输入 ID：", x[:5].tolist())
    print("前 5 个目标 ID：", y[:5].tolist())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    model.train()
    for step in range(500):
        xb, yb = get_batch(x, y, batch_size=32, device=device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 499:
            print(f"step={step:03d}, loss={loss.item():.4f}")

    # 用“小”作为开头，让模型自由生成后续字符。
    start_id = stoi["小"]
    output_ids = model.generate(start_id, max_new_chars=60)
    print("\n生成结果：")
    print(decode(output_ids, itos))

    # 附加思考：
    # 1. 为什么即使训练 loss 很低，模型仍然可能生成不合理的句子？
    # 2. 如果总是使用 argmax 而不是 multinomial，生成结果会有什么变化？
    # 3. 如果想利用前 8 个字符而不是前 1 个字符，模型结构需要怎样改变？


if __name__ == "__main__":
    main()
