# -*- coding: utf-8 -*-
"""
第二章 · 课后小任务：调参实验 —— 隐藏层宽度对准确率的影响

对应课件：《深度学习—第二章》「课后小任务 · 任务二」
把隐藏层从 32 改成 8、128，各训练一次，比较测试集准确率。

运行：python 07_调参实验_隐藏层宽度.py
"""
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

WIDTHS = [4, 8, 32, 128]
EPOCHS = 150


def make_net(width, use_relu=True):
    layers = [nn.Linear(64, width)]
    if use_relu:
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 10))
    return nn.Sequential(*layers)


def run(width, use_relu=True):
    X_raw, y_raw = load_digits(return_X_y=True)
    X = torch.tensor(X_raw, dtype=torch.float32)
    X = (X - X.mean()) / X.std()        # 标准化
    y = torch.tensor(y_raw, dtype=torch.long)
    X_train, y_train, X_test, y_test = X[:1500], y[:1500], X[1500:], y[1500:]

    torch.manual_seed(0)
    model = make_net(width, use_relu)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    params = sum(p.numel() for p in model.parameters())
    return acc, params


def main():
    print('=' * 62)
    print('调参实验：隐藏层宽度 vs 测试集准确率')
    print('=' * 62)
    print(f'结构：64 → 宽度 → 10   每档训练 {EPOCHS} 个 epoch\n')
    print('  隐藏层宽度   参数量    测试集准确率')
    rows = []
    for w in WIDTHS:
        acc, params = run(w, True)
        rows.append((w, params, acc))
        print(f'  {w:>8}   {params:>8}   {acc:>8.2%}')

    print('\n对照实验：去掉 ReLU（多层退化成一层线性模型）')
    for w in [32, 128]:
        acc, params = run(w, False)
        print(f'  宽度 {w} 无激活   {params:>8}   {acc:>8.2%}')

    best = max(rows, key=lambda r: r[2])
    print(f'\n本组最佳：宽度 {best[0]}（准确率 {best[2]:.2%}）')
    print('''
[思考题]（下次课开场分享）

  1. 宽度从 4 → 8 → 32，准确率的变化说明了什么？
     提示：参数太少 → 表达力不够（欠拟合）→ 准确率明显下降。
  2. 宽度从 32 加到 128，参数量翻了两番，准确率提升了多少？
     这就是"容量与代价"的权衡：收益递减，而参数、显存、训练时间都在涨。
  3. 去掉 ReLU 后，准确率几乎没变——为什么？
     提示：去激活后整个网络退化成线性模型（课时一的推导）；
     而手写数字本身接近线性可分，所以线性模型也够用。
     非线性的真正价值，要到更复杂的任务（比如第 3 章的真实图像）才显现出来。
''')


if __name__ == '__main__':
    main()
