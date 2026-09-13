# 第二章 · 课时三 上机实践代码包

> 《深度学习技术与应用》· 授课教师：李家琦
> 对应课件：`深度学习—第二章.pdf`（113 页，三课时）
> 本章主题：**搭建并训练你的第一个神经网络**（两层 FFN，手写数字识别）

## 环境要求

```bash
# Anaconda Prompt 中执行
conda activate dl2026
```

依赖：`torch`、`scikit-learn`、`tensorboard`（缺哪个补哪个：`pip install torch scikit-learn tensorboard`）。
CPU 即可运行，无需显卡。

## 文件清单与使用顺序

| 顺序 | 文件 | 对应课件 | 说明 |
|------|------|----------|------|
| 1 | `01_数据准备.py` | Task 1 · 准备数据 | 加载 sklearn 手写数字，看真实图像，标准化，划分训练/测试集 |
| 2 | `02_搭建网络.py` | Task 2 · 拆解 nn.Module | nn.Module 三件套；打印参数量并手算核对 |
| 3 | `03_训练循环.py` | Task 3 · 五步训练循环 | **核心**：五步舞曲跑通训练（Adam, lr=1e-2, 100 epoch），保存 `model_ch2.pth` |
| 4 | `04_评估.py` | Task 4 · 测试集评估 | 测试集准确率、分数字统计、抽查真实预测 |
| 5 | `05_TensorBoard.py` | Task 5 · TensorBoard | 记录 loss / acc / 权重直方图，浏览器可视化 |
| 6 | `06_课堂练习_训练循环排错.py` | 随堂练习 | 三处高频 bug 判断（先自己答） |
| 7 | `06_参考答案_训练循环排错.py` | — | 练习完成后对答案（勿提前看） |
| 8 | `07_调参实验_隐藏层宽度.py` | 课后小任务 · 任务二 | 隐藏层 4/8/32/128 对照实验（含"去掉 ReLU"对照） |

## 运行方式

所有文件都在 **Anaconda Prompt** 中、且激活 `(dl2026)` 环境后运行：

```bash
cd 上机实践
python 01_数据准备.py
python 02_搭建网络.py
python 03_训练循环.py
python 04_评估.py
```

TensorBoard 需要开两个窗口：

```bash
# 窗口 1：训练并写入日志
python 05_TensorBoard.py

# 窗口 2：启动服务，然后浏览器打开 http://localhost:6006
tensorboard --logdir=runs
```

## 报错锦囊（本章三个高频坑）

| 报错关键字 | 原因 | 解决 |
|-----------|------|------|
| `mat1 and mat2 shapes cannot be multiplied` | 64 维输入喂了 8×8 的原始图 | 先展平：`X = X.reshape(-1, 64)` |
| `expected scalar type Long but found Float` | 标签是 float，`CrossEntropyLoss` 要求整数类别 | `y = y.long()` |
| `'tensorboard' 不是内部或外部命令` | 没装 tensorboard 或没激活环境 | `conda activate dl2026` 后 `pip install tensorboard` |

## 本节验收清单（对应课件「本课时验收清单」）

- [ ] 五个 Task 全部跑通
- [ ] `02_搭建网络.py` 的参数量与手算一致（2410）
- [ ] `04_评估.py` 测试集准确率 ≥ 90%（本配置约 91%~92%）
- [ ] TensorBoard 里看到自己的 loss 曲线与权重直方图
- [ ] 能向同学讲清 `loss.backward()` 里发生了什么（局部梯度 / 链式法则 / 误差倒传）
- [ ] 完成隐藏层宽度对照实验，并把结果写进实验记录

## 作业提交

实验记录（Word/PDF，可用 `实验记录模板.md` 作骨架）下次课前提交课程群收集表。
**报错与解决过程写得越具体，得分越高。**
