#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: LagrangeM.py
@time: 11/11/2019 6:59 PM
'''
from scipy.optimize import fsolve
import sympy
espl=0.0001
a0=-1
def check(a,b,w):
    s = a**2-b-3
    print ("a**2-b-3:",s)
    if s<float(w)/float(2):
        return True
    else:
        return False
def modify(w,v,x1,x2):
    w = max(0,w-2*((x1-2)**2-x2))
    v = v-2*(2*x1-x2-1)
    return (w,v)
def iterf(i,w,v,a):
    b=(-w-v+2*a**2-6+4*a)/6
    a=afs(w,v,b)
    x1=a+2
    x2=b+3
    print ("第%s次迭代：a=%s,b=%s,x1=%s,x2=%s" %(i,a,b,x1,x2))
    wv = modify(w,v,x1,x2)
    w = wv[0]
    v = wv[1]
    print ("w=%s,v=%s" %(w,v))
    #a = asolve(a,w,v,b)
    return (w,v,a)
def asolve(a,w,v,b):
    #af = afs(a,w,v,b)
    return fsolve(afs(a,w,v,b),a0)
def afs(w,v,b):
    res =  fsolve(lambda a: 2*a**3+(1-w-2*b-2)*a-v-2*b, 0)
    #res = sympy.solve('a**3+((1-w)/2-b-1)*a-v/2-b')
    flag = check(res,b,w)
    print ("是否满足：",flag)
    return res