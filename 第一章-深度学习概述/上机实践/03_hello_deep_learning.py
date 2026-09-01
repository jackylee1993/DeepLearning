# -*- coding: utf-8 -*-
"""
03_Hello_Deep_Learning.py —— 第一个深度学习程序
对应 PPT 第 84 页：10 行代码跑通"训练全流程"
任务：让一个 1 个神经元的"网络"学会 OR（或）运算
运行方式（dl2026 环境已激活）：
    python 03_Hello_Deep_Learning.py
预期：预测值从随机逐渐逼近 1（每次运行略有差异属正常）
"""
import torch
import torch.nn as nn

torch.manual_seed(42)  # 固定随机种子，保证结果可复现


def main():
    # ---- 搭建网络：一个最简单的"神经元" ----
    model = nn.Sequential(nn.Linear(2, 1))

    # ---- 数据：OR 运算的 4 种组合（数据 + 答案） ----
    X = torch.tensor([[0., 0.],
                      [0., 1.],
                      [1., 0.],
                      [1., 1.]])
    y = torch.tensor([[0.],
                      [1.],
                      [1.],
                      [1.]])

    # ---- 训练准备：优化器 opt，负责“按梯度更新参数” ----
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # 循环体 = 训练三部曲（①算误差 ②反向传播 ③更新参数）+ 一步清零
    for epoch in range(200):
        loss = ((model(X) - y) ** 2).mean()   # ① 前向计算，得到误差 loss
        loss.backward()                       # ② 反向传播，算出每个参数的梯度
        opt.step()                            # ③ 更新参数（沿梯度反方向走一小步）
        opt.zero_grad()                       # 辅助：清零梯度，否则下一轮会累加（新手必踩坑）
        if epoch % 50 == 0:
            print(f'epoch {epoch:3d}  loss = {loss.item():.4f}')

    # ---- 检验学习成果 ----
    with torch.no_grad():
        for inp in X:
            pred = model(inp.unsqueeze(0)).item()
            print(f'输入 {inp.tolist()} → 预测 {pred:.3f}（答案 {int(inp.sum().item() > 0)}）')

    print()
    print('✔ 恭喜！你已经跑通了第一个深度学习训练。')
    print('  这 10 行里的概念（nn.Linear / loss / backward / step）')
    print('  将在第二章逐行拆解——今天的任务只是"跑起来"。')


if __name__ == '__main__':
    main()
