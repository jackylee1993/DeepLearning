# -*- coding: utf-8 -*-
"""
02_第一个张量.py —— 创建你的第一个张量
对应 PPT 第 81 页：torch.tensor() 把 Python 列表变张量
运行方式（dl2026 环境已激活）：
    python 02_第一个张量.py
"""
import torch


def main():
    # 1) 从 Python 列表创建张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print('张量 x:')
    print(x)
    print('形状 x.shape =', tuple(x.shape), '（2 行 3 列）')
    print()

    # 2) 张量的基本运算
    y = x + 10          # 每个元素加 10
    z = x * 2           # 每个元素乘 2
    m = x @ x.T         # 矩阵乘法（@ 与逐元素乘 * 区分！）
    print('y = x + 10:')
    print(y)
    print('z = x * 2:')
    print(z)
    print('m = x @ x.T（矩阵乘法）形状:', tuple(m.shape))
    print()

    # 3) 常用创建函数
    print('torch.zeros(3, 4) 形状:', tuple(torch.zeros(3, 4).shape), '→ 全零张量')
    print('torch.arange(1, 10) →', torch.arange(1, 10).tolist())
    t = torch.arange(1, 10)
    print('其平均值 =', t.float().mean().item(), '（先 .float() 再 .mean()）')
    print()
    print('✔ 与 NumPy 一行互转：x.numpy() / torch.from_numpy(arr)')


if __name__ == '__main__':
    main()
