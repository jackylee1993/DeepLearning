# -*- coding: utf-8 -*-
"""
06_报错锦囊演示.py —— 亲眼看看常见报错长什么样（可选运行）
对应 PPT 第 87-90 页 / 《上机实践指导书》第八节。
本脚本包含 4 个"故意出错"的函数，逐个运行观察报错信息——
**认识报错是程序员的基本功**。看完请恢复注释，不要留在代码里。
"""
import torch


def error1_wrong_shell():
    """报错①：conda 不是内部或外部命令
    原因：用了普通 CMD/PowerShell，而不是 Anaconda Prompt
    解决：开始菜单搜 "Anaconda Prompt"，以后命令都在那里敲
    （此错误无法在 Python 里复现，是命令行的锅）"""
    print('演示：请在普通 CMD 里敲 conda --version 试试，再用 Anaconda Prompt 对比')


def error2_wrong_env():
    """报错②：ModuleNotFoundError: No module named 'torch'
    原因：装包时装到了别的环境（行首没有 (dl2026)）
    解决：先 conda activate dl2026；或用 python -m pip install"""
    try:
        import torch  # noqa
        print('torch 正常，未复现报错（说明你当前环境是对的）')
    except ModuleNotFoundError as e:
        print('复现报错②：', e)


def error3_version_mix():
    """报错③：python 版本混乱
    检查：行首是否有 (dl2026)？python --version 是否 3.10.x？"""
    import sys
    print('当前 Python:', sys.version.split()[0])


def error4_download_timeout():
    """报错④：CondaHTTPError / 下载超时
    解决三板斧：清华镜像 / pip -i 清华源 / 手机热点
    pip 加速示例（命令行执行，不是 Python）：
        pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple"""
    print('演示：本函数只做说明，不实际下载')


if __name__ == '__main__':
    print('—— 报错锦囊演示（逐个运行，观察输出）——')
    error1_wrong_shell(); print()
    error2_wrong_env(); print()
    error3_version_mix(); print()
    error4_download_timeout()
