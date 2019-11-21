#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: MLEMAP.py
@time: 11/14/2019 3:23 PM
'''
import numpy as np
from scipy.optimize import minimize
#theta= [u,sigma]
# def MLEF(X):
#     fun = lambda theta: -np.multiply.reduce(1/(np.sqrt(2*np.pi)*theta[1]) * np.exp(-0.5*((X-theta[0])/(theta[1])+10E-9)**2))
#     return fun
def MLENLL(X):

    fun = lambda theta: -sum(-0.5*np.log(2*3.1415926*theta[1]**2) - 0.5/theta[1]**2*(X-theta[0])**2)
    return fun

def MAPNLLG2(X):

    fun = lambda theta: -sum(-0.5*np.log(2*3.1415926*theta[1]**2) - 0.5/theta[1]**2*(X-theta[0])**2) + (theta[1]-8)**2/2 + (theta[0]-0.1)**2/2
    return fun

def MAPNLLG1(X):
    fun = lambda theta: -sum(-0.5*np.log(2*3.1415926*theta[1]**2) - 0.5/theta[1]**2*(X-theta[0])**2) + (theta[1]-2)**2/0.01 + (theta[0]-1)**2/0.01
    return fun
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def draw3D(X):
    x = np.arange(0.5, 2.1, 0.01)
    y = np.arange(1.5, 3.2, 0.01)
    x,y = np.meshgrid(x, y)
    #XY = np.concatenate((x,y),axis = 1)
    z =np.zeros(x.shape)
    length = x.shape[0] * x.shape[1]
    for i in range( x.shape[0] ):
        for j in range(x.shape[1]):
           z[i,j] = MLENLL(X)(np.array([x[i,j],y[i,j]]).reshape(-1,1))
    #z = np.array(z).reshape(x.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.jet)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    nm = 8
    #plt.contourf(x, y, z, nm, alpha=0.75, cmap='rainbow')
    C1 = plt.contour(x, y, z, nm, colors='red')
    plt.clabel(C1, inline=1, fontsize=10)
    plt.show()


np.random.seed(1)
for i in range(5):
    n =  10**(i+1)
    X = 2*np.random.randn(n)+1
    theta0 = np.array([0.9,0.9])
    # res = minimize(MLEF(X),theta0,method='BFGS')
    # print("data point:",n)
    # print("x = ", res.x, '\t', "success", res.success, '\t',"function value", res.fun )
    print("data point:", n)
   # res = minimize(MLENLL(X), theta0, method='SLSQP')

    print("MLE")
    res = minimize(MLENLL(X), theta0, method='SLSQP')
    print("x = ", res.x, '\t', "success", res.success, '\t', "function value", res.fun)
    print("MAP")
    res = minimize(MAPNLLG2(X), theta0, method='SLSQP')
    print("x = ", res.x, '\t', "success", res.success, '\t', "function value", res.fun)
    res = minimize(MAPNLLG1(X), theta0, method='SLSQP')
    print("x = ", res.x, '\t', "success", res.success, '\t', "function value", res.fun)
    #draw3D(X)

