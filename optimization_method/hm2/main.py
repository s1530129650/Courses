#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: main.py
@time: 10/29/2019 5:15 PM
'''


import numpy as np
from NewtonMethod import NewtonMethod
from QuasiNewtonMethod import QuasiNewtonMethod
from SteDesMet import SteeDesMethod2 as SteeDesMethod
from myutils import funcRSDP ,plot3D,funcBound,funcRSSDP,funcConvex
from Tao import SDA


# A = np.array([[10,1],[1,-9]])
# b = np.array([-9,-4])
# plot3D(A,b)
dim = 2
x0 = np.random.randn(dim)
threshold = 1e-6

A,b = funcRSDP(dim) # consistend convex
#A,b =  funcBound(dim) # bounded convex
#A,b = funcConvex(dim) # convex
#xstar = -np.dot(np.linalg.inv(A),b)
#x0 = xstar + 1
#fstar = 0.5*np.dot(np.dot(xstar,A),xstar) + np.dot(b,xstar)
#print("f* =  ", fstar)
#x0 = xstar + 0.01 #near
#x0 = -1*np.dot(np.linalg.inv(A),b) + np.random.randn(dim)#waway
#x0 =SteeDesMethod(x0,A,b, threshold, dim) + 10* np.random.randn(dim)
SteeDesMethod(x0,A,b, threshold, dim)
SDA(A,b,x0, threshold)
NewtonMethod(x0,A,b, threshold, dim)
QuasiNewtonMethod(x0,A,b, threshold, dim)