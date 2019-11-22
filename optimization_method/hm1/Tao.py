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
@time: 10/15/2019 1:29 PM
'''
import numpy as np
def CGM(A,b,threshold=10 ** -10):
    k, s = np.shape(b)
    m, n = np.shape(A)
    x_list = []
    z_list = []
    r_list = []
    if m != n:
        print("A不是方阵")
        return
    if m != k:
        print("A,b 维度不匹配")
        return
    if s != 1:
        print("b不是列向量")
        return
    x_ = np.zeros((m, 1))
    r_ = b - np.dot(A, x_)
    d_ = r_
    x_list.append(x_)
    z_list.append(np.dot(x_.T, np.dot(A, x_)) + np.dot(b.T, x_))
    norm_r = np.linalg.norm(r_)
    r_list.append(norm_r)
    while np.linalg.norm(np.dot(A, x_) - b) / np.linalg.norm(b) >= threshold:
        
        a = np.dot(r_.T, r_) / np.dot(d_.T, np.dot(A, d_))
        x = x_ + a * d_
        r = b - np.dot(A, x)
        d = r + np.dot(r.T, r) / np.dot(r_.T, r_) * d_
        x_ = x
        r_ = r
        d_ = d
        x_list.append(x_)
        z_list.append(0.5 * (np.dot(x_.T, np.dot(A, x_)) - np.dot(b.T, x_)).reshape(1))
        norm_r = np.linalg.norm(r_)
        r_list.append(norm_r)
    return x_, x_list, z_list, r_list


import matplotlib.pyplot as plt

def PLOT( r2norm,  func):
    from mpl_toolkits.axes_grid1 import host_subplot
    length = len(r2norm)
    X = np.arange(0, length )
    ax1 = host_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(X, r2norm, 'g*-', label="|r|")
    ax2.plot( X,func, 'bo-', label="f")

    # set digital label

    for a, b in zip(X[-1:], r2norm[-1:]):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=10)

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('|r|')
    ax2.set_ylabel('f')
    #plt.xticks(range(length + 1))
    ax1.legend()  # 显示图例
    plt.title("dim = " + str(dim) + " values of error and objective funtion")

    # 3.result

    print("###############################################")
    print("the result of optimizer is: ", x[-1])
    plt.show()


if __name__=='__main__':
    #generate A,b
    dim = 100
    G = np.random.randint(0,10,size = [dim,dim])
    G = G + 1*np.eye(dim)
    A = G.T @ G
    b = np.random.randint(0,10,size = [dim,1])
    # judge A
    e,v = np.linalg.eig(A)
    if e.all() > 0:
        print("A is RSDP")
    # tao
    x, xlist, zlist, rlist = CGM(A,b)
    # plot
    PLOT( rlist,zlist)


