# -*- coding: utf-8 -*-
"""
第二章 · 课时三 上机实践 · Task 3：五步训练循环
对应课件：《深度学习—第二章》「Task 3 · 五步训练循环」与课时二的「五步舞曲」

运行：python 03_训练循环.py
训练完成后会保存 model_ch2.pth，供 04_评估.py / 05_TensorBoard.py 使用。
"""
import os
import torch
import torch.nn as nn
from sklearn.datasets import load_digits

EPOCHS = 100
LR = 1e-2
SEED = 0
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


def main():
    torch.manual_seed(SEED)
    X_train, y_train, X_test, y_test = get_data()

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print('=' * 62)
    print(f'Task 3 · 五步训练循环（Adam, lr={LR}, {EPOCHS} 个 epoch）')
    print('=' * 62)
    print('五步舞曲：zero_grad → forward → loss → backward → step\n')

    history = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()               # ① 清空上一步的梯度
        logits = model(X_train)             # ② 前向传播
        loss = loss_fn(logits, y_train)     # ③ 算损失
        loss.backward()                     # ④ 反向传播（链式法则）
        optimizer.step()                    # ⑤ 更新参数

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
            history.append((epoch + 1, loss.item(), acc))
            print(f'epoch {epoch + 1:3d} | loss {loss.item():.4f} | 测试集准确率 {acc:.2%}')

    # ---------- 附：用字符画一条 loss 曲线（让你直观看到“学习”在发生） ----------
    print('\nloss 曲线（每 5 个 epoch 一个点，越靠上 loss 越大）：')
    vals = [v for _, v, _ in history]
    lo, hi = min(vals), max(vals)
    for i, (ep, l, acc) in enumerate(history):
        level = 0 if hi - lo < 1e-9 else int(round((l - lo) / (hi - lo) * 14))
        print(f'  epoch {ep:3d} |' + ' ' * (14 - level) + '●' + f'  loss={l:.4f}  acc={acc:.2%}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'\n模型已保存：{MODEL_PATH}')
    print('\n[自检]')
    print('  1) 五步的顺序能背下来吗？（zero_grad → forward → loss → backward → step）')
    print('  2) 删掉 optimizer.zero_grad() 会发生什么？（提示：梯度会累加）')
    print('  3) 下一步：运行 04_评估.py')


if __name__ == '__main__':
    main()
