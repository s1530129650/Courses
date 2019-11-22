#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: ex2.1.py
@time: 11/12/2019 4:12 PM
'''


from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
def drawContour(data_x,data_y,xstar):
    x_arange = np.linspace(-1.0, 3.0, 256)
    y_arange = np.linspace(-1.0, 3.0, 256)
    X, Y = np.meshgrid(x_arange, y_arange)
    Z1 =  X ** 2 + Y ** 2
    Z2= X*X - 1
    #Z2 = Y - X ** 2
    #Z3 = X + Y
    plt.xlabel('x')
    plt.ylabel('y')
    nm =  3
    plt.contourf(X, Y, Z1, nm, alpha=0.75, cmap='gray_r')
    plt.contourf(X, Y, Z2, nm, alpha=0.75, cmap='gray_r')
    #plt.contourf(X, Y, Z3, nm, alpha=0.75, cmap='rainbow')
    C1 = plt.contour(X, Y, Z1, nm, colors='black')
    C2 = plt.contour(X, Y, Z2, nm, colors='blue')
    #C3 = plt.contour(X, Y, Z3, 8, colors='red')
    plt.clabel(C1, inline=1, fontsize=10)
    plt.clabel(C2, inline=1, fontsize=10)
    #plt.clabel(C3, inline=1, fontsize=10)
    plt.plot(data_x, data_y, marker = "*",c = "g",ms ="10")
    plt.scatter(xstar[0], xstar[1], marker=(5, 1), c="r", s=1000)
    #plt.plot(1,0,"*", ms = 10)
    plt.show()
#约束问题p1,控制误差epsilon，罚函数放大系数c,
def func ():
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    return fun
def penalty(sigma_k):
    pfun =  lambda x: sigma_k*(min((x[0]*x[0]-1 ),0))**2
    return pfun
# step2 min p(x, σ) = f(x) + σp(x)yiba σ>0 ,  p(x)yiba 是惩罚项 p(x)yiba = ,得到最优解，这个不难，无约束优化问题最优解很好解
def penalty_func(sigma_k):
    pfun =  lambda x: x[0] ** 2 + x[1] ** 2 + sigma_k*(min((x[0]-1 ),0))**2
    return pfun
if __name__ == "__main__":
    ##step1 初始化x0,惩罚因子 xigema 1, k =1
    epsilon = 10E-5
    c = 10
    x0 = np.array((2.0, 1.0))
    sigma_k = 1
    k = 0
    #res = minimize(penalty_func(sigma_k) , x0, method='SLSQP', )
    #####
    print("x = ",x0,'\t',"success :" + "True",'\t',"fucntion value",func()(x0))
    #判断 σp（x） < epsilon ，xk就是近似最优解，停止，否则σk+1 = cσk ，k = k+1 转step2
    x = x0
    data_x = [x]
    print("penalty(x)",penalty(sigma_k)(x))
    while(True):
        sigma_k = c*sigma_k
        res = minimize(penalty_func(sigma_k), x, method='SLSQP', )
        x = res.x
        print("Iteration time : ",k)
        print("x = ", res.x, '\t', "success", res.success, '\t',
              "function value", res.fun - penalty(sigma_k)(x), "\t", "penalty value :", penalty(sigma_k)(x))
        k = k + 1
        data_x.append(x)
        if penalty(sigma_k)(x) < epsilon:
            break
    data_x = np.array(data_x)
    drawContour(data_x[:, 0], data_x[:, 1], x)

