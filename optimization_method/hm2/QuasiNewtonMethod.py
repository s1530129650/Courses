#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: QuasiNewtonMethod.py
@time: 10/29/2019 1:14 PM
'''
import numpy as  np
from myutils import PlotSub,PlotWithAxis

def QuasiNewtonMethod(x,  A, b ,threshold, dim):
    #parameter set
    maxk = 1e5
    rho = 0.55
    sigma = 0.4

    k = 0
    n = np.shape(x)[0]
    #Hessian maitix
    Hk = np.eye(dim)
    grad = np.dot(A, x) + b
    grad_length = np.linalg.norm(grad)
    f = 1 / 2 * np.dot(np.dot(x, A), x) + np.dot(b, x)
    #init
    data_x = [x]
    data_f = [f]
    data_g = [grad_length]
    # main loop
    while k < maxk:

        if np.linalg.norm(grad) < threshold:
            break
        dk = -1.0*np.dot(Hk,grad)
        # calcualte alpha
        m = 0
        mk = 0
        while (m < 20):
            x_new = x + rho ** m * dk
            newf = 1 / 2 * np.dot(np.dot(x_new, A), x_new) + np.dot(b,x_new)
            oldf = 1 / 2 * np.dot(np.dot(x, A), x) + np.dot(b, x)
            if newf < oldf :
                mk = m
                break
            m = m + 1

        #DFP update

        xk = x + rho ** mk* dk
        sk = xk - x
        yk = np.dot(A,dk)

        if np.dot(sk,yk) > 0:
            Hy = np.dot(Hk,yk)
            sy = np.dot(sk,yk)
            yHy = np.dot(np.dot(yk,Hk),yk)
            #Hk = Hk - np.dot(Hy,Hy)/yHy + np.dot(sk,sk)/sy
            Hk = Hk - 1.0 * Hy.reshape((n, 1)) * Hy / yHy + 1.0 * sk.reshape((n, 1)) * sk / sy
        k += 1
        x = xk
        grad = np.dot(A, x) + b
        grad_length =  np.linalg.norm(grad)
        f = 1 / 2 * np.dot(np.dot(xk, A), xk) + np.dot(b, xk)
        # update
        data_x.append(xk)
        data_f.append(f)
        data_g.append(grad_length)
    # print("len", len(data_x))
    # print("iterate time : ",k)
    # print("optimizer is x = ", xk, "  f = ", f)
    print("optimizer is  f = ", f)
    PlotSub(data_g, data_f, dim, threshold)
    if dim == 2:
        data_x = np.array(data_x)
        PlotWithAxis(data_x[:,0],data_x[:,1], x)


if __name__ == '__main__':
	# set dimension,initial state and threshold
    dim = 10
    x0 = np.zeros(dim)
    threshold = 10e-7
    QuasiNewtonMethod(x0,threshold,dim)