#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: ex1.py
@time: 11/10/2019 3:02 PM
'''
from scipy.optimize import minimize
import numpy as np
e = 1e-10 # 非常接近0的值
fun = lambda x : (x[0] - 0.667) / (x[0] + x[1] + x[2] - 2) # 约束函数
cons = ({'type': 'eq', 'fun': lambda x: x[0] * x[1] * x[2] - 1}, # xyz=1
        {'type': 'ineq', 'fun': lambda x: x[0] - e}, # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[1] - e},
        {'type': 'ineq', 'fun': lambda x: x[2] - e}
       )
x0 = np.array((1.0, 1.0, 1.0)) # 设置初始值
res = minimize(fun, x0, method='SLSQP', constraints=cons)
print('最小值：',res.fun)
print('最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)
