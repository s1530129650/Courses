#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: plot.py
@time: 11/13/2019 9:11 AM
'''
from scipy.optimize import minimize
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


# 目标函数：
def func(args):
    fun = lambda x: 10 - x[0] ** 2 - x[1] ** 2
    return fun


# 约束条件，包括等式约束和不等式约束
def con(args):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0] ** 2},
            {'type': 'eq', 'fun': lambda x: x[0] + x[1]})
    return cons


# 画三维模式图
def draw3D():
    fig = plt.figure()
    ax = Axes3D(fig)
    x_arange = np.arange(-5.0, 5.0)
    y_arange = np.arange(-5.0, 5.0)
    X, Y = np.meshgrid(x_arange, y_arange)
    Z1 = 10 - X ** 2 - Y ** 2
    Z2 = Y - X ** 2
    Z3 = X + Y
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, Z3, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


# 画等高线图
def drawContour():
    x_arange = np.linspace(-3.0, 4.0, 256)
    y_arange = np.linspace(-3.0, 4.0, 256)
    X, Y = np.meshgrid(x_arange, y_arange)
    Z1 = 10 - X ** 2 - Y ** 2
    Z2 = Y - X ** 2
    Z3 = X + Y
    plt.xlabel('x')
    plt.ylabel('y')
    nm =10
    plt.contourf(X, Y, Z1, nm, alpha=0.75, cmap='rainbow')
    plt.contourf(X, Y, Z2, nm, alpha=0.75, cmap='rainbow')
    plt.contourf(X, Y, Z3, nm, alpha=0.75, cmap='rainbow')
    C1 = plt.contour(X, Y, Z1, 8, colors='black')
    C2 = plt.contour(X, Y, Z2, 8, colors='blue')
    C3 = plt.contour(X, Y, Z3, 8, colors='red')
    plt.clabel(C1, inline=1, fontsize=10)
    plt.clabel(C2, inline=1, fontsize=10)
    plt.clabel(C3, inline=1, fontsize=10)
    plt.show()


if __name__ == "__main__":
    args = ()
    args1 = ()
    cons = con(args1)
    x0 = np.array((1.0, 2.0))  # 设置初始值，初始值的设置很重要，很容易收敛到另外的极值点中，建议多试几个值

    # 求解#
    res = minimize(func(args), x0, method='SLSQP', constraints=cons)
    #####
    print(res.fun)
    print(res.success)
    print(res.x)

    #draw3D()
    drawContour()