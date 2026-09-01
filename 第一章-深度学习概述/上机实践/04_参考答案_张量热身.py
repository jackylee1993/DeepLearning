# -*- coding: utf-8 -*-
"""
04_参考答案_张量热身.py —— 随堂练习参考答案（练习完成前请勿直接看）
"""
import torch


def q1():
    x = torch.zeros(3, 4)
    print('第1题 x =', x, sep='\n')
    print('第1题 shape =', tuple(x.shape))


def q2():
    t = torch.arange(1, 10)          # 1..9
    print('第2题 t =', t.tolist())
    mean = t.float().mean()
    print('第2题 平均值 =', mean.item(), '（= 5.0）')


def q3(epochs=2000):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [1.]])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        loss = ((model(X) - y) ** 2).mean()
        loss.backward(); opt.step(); opt.zero_grad()
        if epoch % 500 == 0:
            print(f'  epoch {epoch:4d}  loss = {loss.item():.4f}')
    with torch.no_grad():
        final_loss = loss.item()
        pred = model(torch.tensor([[1., 0.]])).item()
    print(f'第3题（EPOCHS={epochs}）最终 loss = {final_loss:.5f}  预测[1,0] = {pred:.4f}')
    print('观察：相比 200 轮，2000 轮 loss 更小、预测更接近 1；')
    print('但训练越久 ≠ 泛化越好（过拟合风险），第 5 章展开。')


if __name__ == '__main__':
    print('—— 参考答案 ——')
    q1(); print()
    q2(); print()
    print('第3题：')
    q3()
