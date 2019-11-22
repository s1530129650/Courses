#!/usr/bin/env python
#!-*-coding:utf-8 -*-
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence
@contact: *******@qq.com
@site:
@software: PyCharm
@file: myutils.py
@time: 10/29/2019 9:46 AM
'''

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from scipy.stats import ortho_group



def func(dim):
    i = 1
    while True:
        np.random.seed(i)
        i =i+1
        #G = np.random.randint( 0,dim,size =[dim, dim])
        G = np.random.randn(dim, dim)
        A = G.T @ G
        e, v = np.linalg.eig(A)
        if e.all() > 0:
            print("A is RSDP")
            condA = max(e) / min(e)
            print("Conditional number :", condA)
            b = np.random.randint(0, 10, size=[dim])
            # print("A is: ", A)
            # print("***********************************")
            print("eigenvalues: ", e)
            return A,b
def funcRSDP(dim):

    np.random.seed(0)
    A = np.eye(dim)
    for i in np.arange(0,dim):
        A[i,:] = 1
        A[i][i] =i + 1
    m = 1
    A = A+m*np.eye(dim)
    b = np.random.randint(0, dim, size=[dim])
    e, v = np.linalg.eig(A)
    print(A)
    print("##############################")
    print(e)
    condA = max(e) / min(e)
    print("Conditional number : ",   condA)
    return A,b

def funcRSSDP(dim):
    print("A is real-symmetric-semi-define-positive")
    np.random.seed(0)
    A = np.eye(dim)
    for i in np.arange(0, dim):
        A[i, :] = 1
        A[i][i] = i
    # A[i,:] = (np.arange(0,dim))%10 + 1
    # A[:, i] =  A[i,:]
    b = np.random.randint(0, 10, size=[dim])
    e, v = np.linalg.eig(A)
    print(A)
    print("##############################")
    print(e)
    #condA = max(e) / min(e)
    #print("Conditional number : ", condA)
    return A, b

def funcConvex(dim):
    eigvals = 10 * np.abs(np.random.random((dim)))
    A = np.eye(dim)
    for i in range(dim):
        A[i][i] = eigvals[i]
    seed = 1
    U = np.float32(ortho_group.rvs(dim=dim, random_state=seed))
    b = np.random.rand(dim)
    A = U.transpose().dot(A).dot(U)
    e, v = np.linalg.eig(A)
    print(A)
    print("##############################")
    if min(e) != 0:
        condA = max(e) / min(e)
        print("Conditional number : ", condA)
    return A,b
def funcBound(dim):
    print("A is real-symmetric-semi-define-positive")
    np.random.seed(0)
    A = np.eye(dim)
    for i in np.arange(0, dim):

        A[i][i] = (i) % 10+1
    A[i,:] = (np.arange(0,dim))%10 + 1
    # A[:, i] =  A[i,:]
    b = np.random.randint(0, 10, size=[dim])
    e, v = np.linalg.eig(A)
    print(A)
    print("##############################")
    print(e)
    condA = max(e) / min(e)
    print("Conditional number : ", condA)
    return A, b
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

def PlotWithAxis(data_x,data_y,xstar):

    # 绘图
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)

    #plt.title(r'$Gradient \ method - steepest \ descent \ method$')
    #plt.plot(data_x, data_y, label=r'$f(x_1,x_2)=x_1^2+2 \cdot x_2^2-2 \cdot x_1 \cdot x_2-2 \cdot x_2$')
    #plt.title(r'$Gradient \ method - steepest \ descent \ method$')
    plt.plot(data_x, data_y, label=r'$f(x_1,x_2)=Ax + b$')
    #plt.grid(True, color="r")
    plt.legend()
    plt.scatter(xstar[0], xstar[1],marker=(5,1), c="r", s=1000)
    #plt.grid(True,color='r', linestyle ='-', linewidth = 2)
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.show()

def PlotSub(r,func,dim,threshold):#residual error; function value, dimension, threshold
    plt.figure()
    X = np.arange(0, len(r))
    length = len(X)
    # fig1
    ax1 = plt.subplot(3, 1, 1)
    if len(X) < 50:
        plt.plot(X, r, 'g*-')
    else:
        plt.plot(X, r, 'g-')
    ax1.set_title(" Error value   " + "(dim = " +str(dim) +", " +"threshold= " +str(threshold) +")")
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('grad value')
    #plt.xticks(range(length + 1))

    # fig2
    ax2 = plt.subplot(3, 1, 3)
    if len(X) < 50:
        plt.plot(X, func, 'bo-')
    else:
        plt.plot(X, func, 'b-')
    ax2.set_title("The objection function value   " + "(dim = " +str(dim)+", " +"threshold = " +str(threshold) +")")
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('fun value')
    #plt.xticks(range(length + 1))

    plt.show()

from mpl_toolkits.mplot3d import Axes3D
def plot3D(A,b):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
    y = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
    X, Y = np.meshgrid(x, y)  # 网格的创建，这个是关键

    Z = A[0,0]*X**2 + (A[0,1]+A[1,0])*X*Y + A[1,1]*Y**2 + b[0]*X +b[1]*Y

    Z = np.array(Z)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
