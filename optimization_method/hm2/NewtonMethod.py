#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: NewtonMethod.py
@time: 10/29/2019 9:46 AM
'''
import numpy as np
from myutils import PlotSub,PlotWithAxis

def NewtonMethod(x,A,b, threshold,dim):#initial state ,threshold,dimension
    grad_vec = np.dot(A,x) + b
    grad_length = np.linalg.norm(grad_vec)
    k = 0
    f = 1 / 2 * np.dot(np.dot(x,A),x) + np.dot(b,x)
    data_x = [x]
    data_f = [f]
    data_g = [grad_length]
    delta_norm = 1
    while delta_norm  > threshold:
        k += 1
        grad = grad_vec
        # iterate
        alpha = np.linalg.inv(A)
        #print(alpha)
        delta = - np.dot(alpha, grad)
        x = x + delta
        #gradient
        grad_vec = np.dot(A, x) + b
        grad_length = np.linalg.norm(grad_vec)
        delta_norm = np.linalg.norm(delta)
        f =1 / 2 * np.dot(np.dot(x,A),x) + np.dot(b,x)
        #uodate
        data_x.append(x)
        data_f.append(f)
        data_g.append(grad_length)
    # print("len",len(data_x))
    # print(k)
    # print("optimizer is x = ",x,"  f = ",f)
    #PlotWithAxis(data_x,data_y)
    print("optimizer is  f = ", f)
    PlotSub(data_g, data_f, dim, threshold)
    if dim == 2:
        data_x = np.array(data_x)
        PlotWithAxis(data_x[:,0],data_x[:,1], x)

if __name__ == '__main__':
	# set dimension,initial state and threshold

    dim = 2
    x0 = np.zeros(dim)
    threshold = 10e-7
    NewtonMethod(x0,threshold,dim)

