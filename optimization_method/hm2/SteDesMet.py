#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: SteDesMet.py
@time: 10/28/2019 10:38 PM
'''
import numpy as  np
from myutils import PlotSub,PlotWithAxis


def SteeDesMethod(x, A, b , threshold,dim):#initial state ,threshold,dimension
    grad_vec = np.dot(A,x) + b
    grad_length = np.linalg.norm(grad_vec)
    k = 0
    f = 1 / 2 * np.dot(np.dot(x,A),x) + np.dot(b,x)
    data_x = [x]
    data_f = [f]
    data_g = [grad_length]
    delta = 1
    while  delta >= threshold:
        k += 1
        grad = -grad_vec
        # iterate
        alpha = np.dot(grad,grad) / np.dot(np.dot(grad,A),grad)
        #print(alpha)
        x = x + alpha* grad
        delta = alpha * grad_length
        #gradient
        grad_vec = np.dot(A, x) + b
        grad_length = np.linalg.norm(grad_vec)
        f =1 / 2 * np.dot(np.dot(x,A),x) + np.dot(b,x)
        #update
        data_x.append(x)
        data_f.append(f)
        data_g.append(grad_length)
    #print("len",len(data_x))
   # print(k)
    #print("optimizer is x = ",x,"  f = ",f)
    #print("optimizer is :", data_x[3])
    #PlotWithAxis(data_x,data_y)
    print("f",data_f)
    #print("f", data_f)
    PlotSub(data_g, data_f, dim, threshold)


def SteeDesMethod2(x, A, b , threshold,dim):#initial state ,threshold,dimension
    grad_vec = np.dot(A,x) + b
    grad_length = np.linalg.norm(grad_vec)
    k = 0
    f = 1 / 2 * np.dot(np.dot(x,A),x) + np.dot(b,x)
    data_x = [x]
    data_f = [f]
    data_g = [grad_length]
    delta = 1
    while   grad_length >= threshold:
        k += 1
        grad = -grad_vec
        # iterate
        alpha = np.dot(grad,grad) / np.dot(np.dot(grad,A),grad)
        #print(alpha)
        x = x + alpha* grad
        delta = alpha * grad_length
        #gradient
        grad_vec = np.dot(A, x) + b
        grad_length = np.linalg.norm(grad_vec)
        f =1 / 2 * np.dot(np.dot(x,A),x) + np.dot(b,x)
        #update
        data_x.append(x)
        data_f.append(f)
        data_g.append(grad_length)
    #print("f", data_f)
    #print("g", data_g)
    # print("len",len(data_x))
    # print(k)
    # print("optimizer is x = ",x,"  f = ",f)
    print("optimizer is  f = ", f)
    #print("optimizer is :", data_x[3])
    #PlotWithAxis(data_x,data_y)
    PlotSub(data_g, data_f, dim, threshold)
    if dim == 2:
        data_x = np.array(data_x)
        PlotWithAxis(data_x[:,0],data_x[:,1], x)

    return data_x[data_f.index(min(data_f))]

if __name__ == '__main__':
	# set dimension,initial state and threshold

    dim = 10
    x0 = np.zeros(dim)
    threshold = 10e-7
    SteeDesMethod(x0,threshold,dim)
