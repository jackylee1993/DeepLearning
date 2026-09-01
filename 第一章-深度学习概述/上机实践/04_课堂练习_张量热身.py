# -*- coding: utf-8 -*-
"""
04_课堂练习_张量热身.py —— 随堂练习：Tensor 热身三题
对应 PPT 第 93 页。先把 TODO 全部补完，再运行；答案见 04_参考答案_张量热身.py
运行方式（dl2026 环境已激活）：
    python 04_课堂练习_张量热身.py
"""
import torch


def q1():
    """第 1 题：创建形状为 (3, 4) 的全零张量，并打印 shape"""
    # TODO: 用 torch.zeros 创建，赋值给 x
    x = None  # ← 改成你的答案
    print('第1题 shape =', tuple(x.shape) if x is not None else '未完成')


def q2():
    """第 2 题：创建 1~9 的张量，输出平均值（提示：.float().mean()）"""
    # TODO: 用 torch.arange 创建 t，然后求平均
    t = None  # ← 改成你的答案
    mean = None  # ← 改成你的答案
    print('第2题 平均值 =', mean.item() if mean is not None else '未完成')


def q3():
    """第 3 题：把第 84 页 Hello Deep Learning 的训练轮数改成 2000，
    观察预测值变化（训练越久 ≠ 一定越好，第 5 章见分晓）"""
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [1.]])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    EPOCHS = 200  # TODO: 改成 2000，对比最终 loss 和预测值
    for epoch in range(EPOCHS):
        loss = ((model(X) - y) ** 2).mean()
        loss.backward(); opt.step(); opt.zero_grad()
        if epoch % 500 == 0:
            print(f'  epoch {epoch:4d}  loss = {loss.item():.4f}')
    with torch.no_grad():
        print(f'第3题（EPOCHS={EPOCHS}）最终 loss = {loss.item():.4f}',
              '预测[1,0] =', model(torch.tensor([[1., 0.]])).item())


if __name__ == '__main__':
    print('—— 张量热身三题 ——')
    q1(); print()
    q2(); print()
    print('第3题：')
    q3()
