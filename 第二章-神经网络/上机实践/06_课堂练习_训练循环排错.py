# -*- coding: utf-8 -*-
"""
第二章 · 课时三 随堂练习：训练循环排错（不含答案）

三处都是真实作业里的高频错误。请先自己想，填写下面的答案，
再运行本脚本自检（对比 06_参考答案_训练循环排错.py 之前，先别偷看）。

运行：python 06_课堂练习_训练循环排错.py
"""

print('=' * 62)
print('随堂练习 · 训练循环排错')
print('=' * 62)

print('''
【错误 1】下面五行训练代码，backward 和 step 的顺序写反了：

    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.step()        # ← 顺序错了
    loss.backward()

问题：这样写会发生什么？为什么？

【错误 2】忘记写 optimizer.zero_grad()：

    for epoch in range(50):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

问题：loss 曲线会怎么变化？

【错误 3】评估时忘了加 with torch.no_grad()：

    logits = model(X_test)
    preds = logits.argmax(dim=1)

问题：最直接的影响是什么？
''')

# ---------------- 请把你的答案写在下面（单项选择题） ----------------
# 每题选一个：'A' / 'B' / 'C'
answer1 = ''    # 错误 1
answer2 = ''    # 错误 2
answer3 = ''    # 错误 3

KEY = {
    '1': ('A', '用上一步的旧梯度更新参数，等于白更新；正确顺序是先 backward 再 step'),
    '2': ('B', '梯度会逐批累加，loss 曲线剧烈震荡、难以收敛（不清零 → 越滚越大）'),
    '3': ('C', '会一直构建/保留计算图，显存（内存）占用飙升，还可能误改参数'),
}
OPTIONS = {
    '1': ['A. 等于用旧梯度更新一步，训练乱套', 'B. 没有影响', 'C. 程序直接报错退出'],
    '2': ['A. 完全没影响', 'B. 梯度累加 → loss 剧烈震荡不收敛', 'C. loss 一定变成 0'],
    '3': ['A. 准确率会变高', 'B. 预测结果会错', 'C. 显存/内存占用明显变大'],
}

if __name__ == '__main__':
    given = {'1': answer1.strip().upper(), '2': answer2.strip().upper(), '3': answer3.strip().upper()}
    ok = 0
    for k in ['1', '2', '3']:
        print(f'--- 第 {k} 题 ---')
        for o in OPTIONS[k]:
            print('   ' + o)
        if not given[k]:
            print('   （你还没有填写 answer%s）\n' % k)
            continue
        right, why = KEY[k]
        if given[k] == right:
            ok += 1
            print(f'   你的答案：{given[k]}  ✓ 正确')
        else:
            print(f'   你的答案：{given[k]}  ✗ 再想想')
        print(f'   解析：{why}\n')
    print(f'得分：{ok}/3')
    if ok < 3:
        print('请修改脚本顶部的 answer1 / answer2 / answer3 后重新运行。')
    else:
        print('全对！现在打开 06_参考答案_训练循环排错.py 对照一遍，然后把这三点写进实验记录。')
