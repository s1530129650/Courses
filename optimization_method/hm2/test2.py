# !/usr/bin/python
# coding: utf-8

# 有交互效果，可以在交互式编译器中观察每一步的执行过程

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21)

plt.plot(x, x * 2)

# True 显示网格
# linestyle 设置线显示的类型(一共四种)
# color 设置网格的颜色
# linewidth 设置网格的宽度 
plt.grid(True, linestyle="-.", color="r", linewidth="3")


plt.show()
