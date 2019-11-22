#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: Tao.py
@time: 10/30/2019 7:09 AM
'''
import numpy as np
from myutils import PlotSub

def SDA(A,b,start,threshold=10 ** -6):

    x_list = []
    z_list = []
    r_list = []

    x_ = start
    d_ = -b - np.dot(A,x_)
    x_list.append(x_)
    z_list.append(0.5*np.dot(x_.T,np.dot(A,x_)) + np.dot(b.T,x_))
    norm_r = np.linalg.norm(d_)
    r_list.append(norm_r)
    while np.linalg.norm(np.dot(A, x_) + b) >= threshold:
        a = np.dot(d_.T,d_)/np.dot(d_.T,np.dot(A,d_))
        x_ = x_ + a*d_
        d_ = -b - np.dot(A,x_)
        x_list.append(x_)
        z_list.append(0.5*np.dot(x_.T,np.dot(A,x_)) + np.dot(b.T,x_))
        norm_r = np.linalg.norm(d_)
        r_list.append(norm_r)
    #print("optimizer is x = ", x_, "  f = ", z_list[-1])
    PlotSub(r_list, z_list, len(x_), threshold)
    #print("f", z_list)
    #print("g", r_list)
    print("optimizer is  f = ", z_list[-1])
    #return x_,x_list,z_list,r_list
def create_A_b(dimension,matrix_type):
    eigvals = 10*np.abs(np.random.random((dimension)))
    A = np.eye(dimension)
    m = abs(np.random.randint(10))
    print(m)
    for i in range(dimension):
        A[i][i] = eigvals[i]
    seed = 1
    U= np.float32(ortho_group.rvs(dim=dimension, random_state=seed))
    b = np.random.rand(dimension, 1)
    A = U.transpose().dot(A).dot(U)
    if matrix_type == "convex":
        return A,b
    if matrix_type == "consistently convex":
        return A+m*np.eye(dimension),b,m
    if matrix_type == "bounded convex":
        return