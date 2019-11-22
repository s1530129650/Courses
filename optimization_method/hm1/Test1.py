#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: Test1.py
@time: 10/15/2019 1:37 PM
'''

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import host_subplot
dim = 500
#1. generate maxtix A and a vector b
# A = GT*G
# f(x) = 1/2 x(T)Ax - b(T)x

# Generate a random dim*dim matrix
G = np.random.randint(0,10,size = [dim,dim])
# obtain strictly diaginally dominant matrix
G = G + dim*np.eye(dim)
# Multiply by its tranpose
A = G.T @ G
e,v = np.linalg.eig(A)

b = np.random.randint(0,10,size = [dim,1])
#print("A is: ",A  )
#print("eigenvalues: " ,e)
#print("const vector b:" ,b)
#print("x* = A-1b = ",np.linalg.inv(A)@b)
#2. init  & iterate
# r(k) = b - Ax(k)
# d(k) = r(k) +  |r(k)|/|r(k-1)|*d(k-1)
# α(k) = |r(k)|/(d(k)(T)Ad(k))

x = np.zeros([dim+2,dim])
r = np.zeros([dim+1,dim])
d = np.zeros([dim+1,dim])
alpha =  np.zeros([dim+1,1])
func =  np.zeros([dim+1,1])
'''
x = np.zeros([6*dim+1,dim])
r = np.zeros([6*dim,dim])
d = np.zeros([6*dim,dim])
alpha =  np.zeros([6*dim,1])
func =  np.zeros([6*dim,1])
'''

for k in range(0, dim+1):
    r[k] = b.T - A @ x[k]
    if k == 0:
        d[k] =  r[k]
    else:
        d[k] = r[k] + np.linalg.norm(r[k]) ** 2 / np.linalg.norm(r[k-1])**2 * d[k-1]
    alpha[k] =np.linalg.norm(r[k],2) **2  / (d[k].T @ A @ d[k])
    x[k+1] =  x[k] + alpha[k]*d[k]
    func[k] = 1 / 2 * x[k].T @ A @ x[k] - b.T @ x[k]
    print("r()k ",np.linalg.norm(r[k] )  )
    if np.linalg.norm(r[k] ) < 10**-8:
        length = k;
        print("length",length)
        break

r2norm = np.linalg.norm(r[0:length+1,],2,1)
print("###############################################")
#print("2 norm of r is : ",r2norm)
#print("the values of object function",func)

X = np.arange(0,length + 1)
ax1 =  host_subplot(111)
ax2 = ax1.twinx()
ax1.plot(X, r2norm, 'g*-',label = "|r|")
ax2.plot(X, func[0:length+1], 'bo-',label = "f")

# set digital label

for a, b in zip(X[-1:], r2norm[-1:]):
    plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=10)



ax1.set_xlabel('epoch')
ax1.set_ylabel('|r|')
ax2.set_ylabel('f')
plt.xticks(range(length+1))
ax1.legend() # 显示图例
plt.title("dim = "+str(dim)+" values of error and objective funtion")

# 3.result
print("x: ",x)
print("###############################################")
print("the result of optimizer is: ",x[-1])
plt.show()
