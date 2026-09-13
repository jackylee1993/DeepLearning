# -*- coding: utf-8 -*-
"""
第二章 · 课时三 上机实践 · Task 2：搭建网络
对应课件：《深度学习—第二章》「Task 2 · 拆解 nn.Module：三件套」

运行：python 02_搭建网络.py
"""
import torch
import torch.nn as nn


# ---------- nn.Module 三件套 ----------
# ① 继承 nn.Module
class Net(nn.Module):
    def __init__(self):
        super().__init__()                      # ② 初始化基类
        self.fc1 = nn.Linear(64, 32)            # 隐藏层：64 维输入 → 32 个神经元
        self.act = nn.ReLU()                    # 激活函数：给网络非线性
        self.out = nn.Linear(32, 10)            # 输出层：32 → 10 个类别

    def forward(self, x):                       # ③ 定义前向传播路径
        return self.out(self.act(self.fc1(x)))


if __name__ == '__main__':
    print('=' * 62)
    print('Task 2 · 搭建网络：64 → 32 → 10')
    print('=' * 62)

    torch.manual_seed(0)
    model = Net()
    print(model)

    # 数一数可学习参数
    total = sum(p.numel() for p in model.parameters())
    manual = 64 * 32 + 32 + 32 * 10 + 10
    print(f'\n可学习参数总量：{total}')
    print(f'手算核对：64×32 + 32 = {64 * 32 + 32}（fc1 的 W 和 b）')
    print(f'          32×10 + 10 = {32 * 10 + 10}（out 的 W 和 b）')
    print(f'          合计 = {manual}  →  与程序输出{"一致" if total == manual else "不一致（请检查）"}')

    # 逐层看形状：一次前向传播
    x = torch.randn(5, 64)                      # 假装 5 个样本
    z1 = model.fc1(x)
    a1 = model.act(z1)
    y = model.out(a1)
    print('\n一次前向传播的形状变化：')
    print(f'  输入 x        {tuple(x.shape)}')
    print(f'  fc1 加权求和   {tuple(z1.shape)}   ← z = x @ W.T + b')
    print(f'  ReLU 激活      {tuple(a1.shape)}   ← 负数清零')
    print(f'  输出 logits    {tuple(y.shape)}   ← 10 个类别的“打分”')

    print('\n[自检]')
    print('  1) 三件套分别是哪三件？（继承 / 定义层 / 写 forward）')
    print('  2) fc1 和课时一的“一个人工神经元”是什么关系？')
    print('  3) 下一步：运行 03_训练循环.py')
