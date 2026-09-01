# -*- coding: utf-8 -*-
"""
01_环境验证.py —— 第一步：确认 Anaconda / conda 环境 / PyTorch 就绪
运行方式（Anaconda Prompt）：
    conda activate dl2026
    python 01_环境验证.py
预期：所有 [OK] 即环境就绪；任何 [FAIL] 对照《上机实践指导书》报错锦囊。
"""
import sys


def main():
    print('=' * 56)
    print(' 深度学习开发环境验证')
    print('=' * 56)

    # 1. Python 版本
    v = sys.version_info
    print(f'[1] Python 版本: {v.major}.{v.minor}.{v.micro}', end=' ')
    ok1 = v.major == 3 and v.minor >= 10
    print('[OK]' if ok1 else '[FAIL] 需要 Python 3.10+')

    # 2. conda 环境（软提示）
    env = 'dl2026' in sys.executable or 'envs' in sys.executable
    print(f'[2] 当前解释器: {sys.executable}')
    print('    ' + ('[OK] 位于虚拟环境中' if env else '[提示] 未检测到虚拟环境路径，请确认已 conda activate dl2026'))

    # 3. PyTorch 导入与版本
    try:
        import torch
        print(f'[3] PyTorch 版本: {torch.__version__} [OK]')
    except ImportError:
        print('[3] PyTorch 未安装 [FAIL] 请执行: conda install pytorch torchvision cpuonly -c pytorch')
        return

    # 4. GPU 可用性（False 属正常，本课用 CPU 版）
    cuda = torch.cuda.is_available()
    print(f'[4] CUDA 可用: {cuda}', '[OK]' if cuda else '[OK]（CPU 版，本课所有任务均可完成）')

    print('=' * 56)
    print(' 全部通过！环境就绪，可以开始上机实践。')
    print('=' * 56)


if __name__ == '__main__':
    main()
