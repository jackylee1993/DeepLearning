# -*- coding: utf-8 -*-
"""
第二章 · 课时三 上机实践 · Task 1：准备数据
对应课件：《深度学习—第二章》「Task 1 · 准备数据：sklearn 手写数字（小规模）」

运行：python 01_数据准备.py
"""
import torch
from sklearn.datasets import load_digits

print('=' * 62)
print('Task 1 · 准备数据：sklearn 手写数字（8×8 灰度图，64 维）')
print('=' * 62)

# ---------- 1. 加载数据 ----------
X_raw, y_raw = load_digits(return_X_y=True)
print(f'样本数：{X_raw.shape[0]}    每张图展平后的特征维数：{X_raw.shape[1]}')
print('图像尺寸：8 × 8 = 64 维（每个像素一个灰度值，0=黑，16=白）')
print(f'类别：{sorted(set(y_raw.tolist()))}')

# ---------- 2. 看一眼真实图像（用字符画出来） ----------
CHARS = ' .:-=+*#%@'


def show(img):
    for row in img:
        print('    ' + ''.join(CHARS[min(9, round(v / 16 * 9))] for v in row))


print('\n第 0 个样本（标签 = %d）的 8×8 像素：' % y_raw[0])
show(X_raw[0].reshape(8, 8))

# ---------- 3. 转成 PyTorch 张量 ----------
# 特征用 float32（网络计算需要浮点）；标签用 long（CrossEntropyLoss 要求整数类别）
X = torch.tensor(X_raw, dtype=torch.float32)
X = (X - X.mean()) / X.std()          # 标准化：把像素值拉到相近尺度（训练更稳）
y = torch.tensor(y_raw, dtype=torch.long)
print(f'\n张量形状：X = {tuple(X.shape)}   dtype = {X.dtype}')
print(f'          y = {tuple(y.shape)}   dtype = {y.dtype}')

# ---------- 4. 划分训练集 / 测试集 ----------
X_train, X_test = X[:1500], X[1500:]
y_train, y_test = y[:1500], y[1500:]
print(f'\n训练集：{tuple(X_train.shape)}    测试集：{tuple(X_test.shape)}')
print('测试集模型训练时从没见过 —— 它才是检验“学会没有”的标准。')

# ---------- 5. 基准线：随机猜 ----------
n_cls = int(y.max()) + 1
print(f'\n随机猜的准确率约 {1 / n_cls:.0%}。我们的目标：测试集 90% 以上。')

print('\n[自检]')
print('  1) X 是 float32、y 是 long 吗？')
print('  2) 你能说清 64 这个数字是怎么来的吗？')
print('  3) 下一步：运行 02_搭建网络.py')
