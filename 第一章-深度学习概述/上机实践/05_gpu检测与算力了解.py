# -*- coding: utf-8 -*-
"""
05_GPU检测与算力了解.py —— 认识你的"燃料"：CPU vs GPU
对应 PPT 第 77-79 页。CPU 版返回 False 属正常，不影响本学期任务。
运行方式（dl2026 环境已激活）：
    python 05_GPU检测与算力了解.py
"""
import time
import torch


def bench(device, n=2000):
    """粗测：n×n 矩阵乘法耗时"""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = a @ b
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / 5


def main():
    print('=' * 56)
    print(' GPU / 算力检测')
    print('=' * 56)

    has_cuda = torch.cuda.is_available()
    print('[1] torch.cuda.is_available() =', has_cuda)
    if has_cuda:
        print('[2] 显卡型号:', torch.cuda.get_device_name(0))
        print('[3] 显存总量: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    else:
        print('[2] 无可用 GPU（CPU 版 PyTorch 或无 NVIDIA 独显）')
        print('    → 本学期前半程完全够用；免费云端 GPU 见指导书"三条路线"')

    print()
    print(f'正在跑基准测试（2000×2000 矩阵乘法 × 5 次）…')
    t_cpu = bench('cpu')
    print(f'  CPU 耗时: {t_cpu*1000:7.1f} ms')
    if has_cuda:
        t_gpu = bench('cuda')
        print(f'  GPU 耗时: {t_gpu*1000:7.1f} ms   加速比: {t_cpu/t_gpu:.1f}x')
        print()
        print('✔ 体会一下：深度学习 = 海量矩阵运算 = GPU 的主场')
    else:
        print()
        print('✔ CPU 也能跑本课全部实验；想体验 GPU 加速可用 Colab/Kaggle/AI Studio')


if __name__ == '__main__':
    main()
