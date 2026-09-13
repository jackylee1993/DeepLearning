# -*- coding: utf-8 -*-
"""
第二章 · 课时三 上机实践 · Task 5：TensorBoard 可视化
对应课件：《深度学习—第二章》「Task 5 · TensorBoard：让训练过程可视化」

运行：python 05_TensorBoard.py

前置：pip install tensorboard
运行后，再开一个命令行窗口执行：
    tensorboard --logdir=runs
然后浏览器打开 http://localhost:6006
"""
import os
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise SystemExit('未安装 tensorboard。请先执行：pip install tensorboard')

EPOCHS = 100
LR = 1e-2
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'ch2_first_net')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.act = nn.ReLU()
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        return self.out(self.act(self.fc1(x)))


def main():
    X_raw, y_raw = load_digits(return_X_y=True)
    X = torch.tensor(X_raw, dtype=torch.float32)
    X = (X - X.mean()) / X.std()          # 标准化：把各像素拉到相近尺度
    y = torch.tensor(y_raw, dtype=torch.long)
    X_train, y_train, X_test, y_test = X[:1500], y[:1500], X[1500:], y[1500:]

    torch.manual_seed(0)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(LOG_DIR)

    print('=' * 62)
    print('Task 5 · TensorBoard：把训练过程画出来')
    print('=' * 62)
    print(f'日志目录：{LOG_DIR}')
    print(f'配置：Adam(lr={LR})，{EPOCHS} 个 epoch\n')

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (model(X_test).argmax(dim=1) == y_test).float().mean()

        writer.add_scalar('train/loss', loss.item(), epoch)   # loss 曲线
        writer.add_scalar('train/acc', acc.item(), epoch)     # 准确率曲线
        for name, p in model.named_parameters():
            writer.add_histogram(name, p.detach(), epoch)     # 权重分布随训练演化

        if (epoch + 1) % 20 == 0:
            print(f'epoch {epoch + 1:3d} | loss {loss.item():.4f} | acc {acc.item():.2%}')

    writer.close()
    print('\n日志写入完毕。现在打开一个新的命令行窗口，执行：')
    print('    tensorboard --logdir=runs')
    print('然后浏览器打开 http://localhost:6006')
    print('\n你要看的三个面板：')
    print('  SCALARS  → train/loss 稳定下降、train/acc 稳定上升')
    print('  HISTOGRAMS / DISTRIBUTIONS → 各层权重分布逐渐“分化”')
    print('\n[自检] 把 loss 曲线和准确率曲线截图，写进实验记录。')


if __name__ == '__main__':
    main()
