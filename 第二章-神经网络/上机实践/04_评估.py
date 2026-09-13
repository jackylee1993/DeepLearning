# -*- coding: utf-8 -*-
"""
第二章 · 课时三 上机实践 · Task 4：测试集评估
对应课件：《深度学习—第二章》「Task 4 · 测试集评估：它真的学会了吗」

运行：python 04_评估.py
若还没有训练好的模型，本脚本会自动训练一次（约 10 秒）。
"""
import os
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_ch2.pth')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.act = nn.ReLU()
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        return self.out(self.act(self.fc1(x)))


def get_data():
    X_raw, y_raw = load_digits(return_X_y=True)
    X = torch.tensor(X_raw, dtype=torch.float32)
    X = (X - X.mean()) / X.std()        # 标准化：把各像素拉到相近尺度
    y = torch.tensor(y_raw, dtype=torch.long)
    return X[:1500], y[:1500], X[1500:], y[1500:]


def train_if_needed(model, X_train, y_train):
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f'已加载模型：{MODEL_PATH}\n')
        return
    print('未找到 model_ch2.pth，先自动训练一次（约 10 秒）……\n')
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()


def main():
    _, _, X_test, y_test = get_data()
    X_train, y_train, _, _ = get_data()
    model = Net()
    train_if_needed(model, X_train, y_train)

    print('=' * 62)
    print('Task 4 · 测试集评估：它真的学会了吗？')
    print('=' * 62)

    model.eval()                        # 切换到推理模式
    with torch.no_grad():               # 评估不需要梯度，省一半内存
        logits = model(X_test)
        preds = logits.argmax(dim=1)    # 取 10 个类别里分数最高的
        acc = (preds == y_test).float().mean().item()

    print(f'测试集样本数：{len(y_test)}')
    print(f'测试集准确率：{acc:.2%}')
    print('随机猜的基准：10%\n')

    # ---------- 逐类准确率：看看哪几个数字容易认错 ----------
    print('分数字统计（测试集）：')
    print('  数字   样本数   认对数   准确率')
    for d in range(10):
        mask = y_test == d
        n = int(mask.sum())
        if n == 0:
            continue
        c = int((preds[mask] == d).sum())
        print(f'   {d:>3}   {n:>6}   {c:>6}   {c / n:>6.1%}')

    # ---------- 看几个真实预测：字符画 + 预测/真值 ----------
    CHARS = ' .:-=+*#%@'
    X_raw, _ = load_digits(return_X_y=True)
    print('\n抽查前 6 个测试样本（✦ = 真实标签，→ = 模型预测）：')
    for i in range(6):
        img = X_raw[1500 + i].reshape(8, 8)
        print(f'\n  样本 #{i + 1}  真值 {int(y_test[i])}  预测 {int(preds[i])}  '
              f'{"✓" if preds[i] == y_test[i] else "✗"}')
        for row in img:
            print('    ' + ''.join(CHARS[min(9, round(v / 16 * 9))] for v in row))

    print('\n[思考] 训练集上学得好不算本事，测试集上好才是泛化 —— 这是第一章讲的原则。')
    print('[自检]')
    print('  1) 为什么评估要加 with torch.no_grad()？')
    print('  2) 如果测试集准确率只有 85%，你会先查什么？')
    print('  3) 下一步：运行 05_TensorBoard.py，把训练过程画出来')


if __name__ == '__main__':
    main()
