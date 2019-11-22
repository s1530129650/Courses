#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: ex2.py
@time: 11/12/2019 2:36 PM
'''

from scipy.optimize import minimize
import numpy as np
#约束问题p1,控制误差epsilon，罚函数放大系数c,
from matplotlib import pyplot as plt
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

def func ():
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    return fun
def penalty(sigma_k):
    pfun =  lambda x: sigma_k*((x[0] + x[1] -2)**2 )
    return pfun
# step2 min p(x, σ) = f(x) + σp(x)yiba σ>0 ,  p(x)yiba 是惩罚项 p(x)yiba = ,得到最优解，这个不难，无约束优化问题最优解很好解
def penalty_func(sigma_k):
    pfun =  lambda x: x[0] ** 2 + x[1] ** 2 + sigma_k*((x[0] + x[1] -2)**2 )
    return pfun
if __name__ == "__main__":
    ##step1 初始化x0,惩罚因子 xigema 1, k =1
    epsilon = 10E-5
    c = 10
    x0 = np.array((1.0, 2.0))
    sigma_k = 1
    k = 0
    #res = minimize(penalty_func(sigma_k) , x0, method='SLSQP', )
    #####
    print("x = ",x0,'\t',"success :" + "True",'\t',"fucntion value",func()(x0))

    #判断 σp（x） < epsilon ，xk就是近似最优解，停止，否则σk+1 = cσk ，k = k+1 转step2
    x = x0
    #print("penalty(x)",penalty(sigma_k)(x))
    while(penalty(sigma_k)(x) > epsilon):
        sigma_k = c*sigma_k
        res = minimize(penalty_func(sigma_k), x, method='SLSQP', )
        x = res.x
        print("Iteration time : ",k)
        print(  "x = ",res.x,'\t',"success", res.success, '\t',
                "function value",res.fun - penalty(sigma_k)(x),"\t","penalty value :",penalty(sigma_k)(x))
        k = k + 1




