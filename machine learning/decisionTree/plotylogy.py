#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: plotylogy.py
@time: 11/19/2019 8:49 AM
'''
from matplotlib import pyplot as plt
import  numpy as np
p= np.linspace(0.01, 1.0001, 5000)
Ent =  -p*np.log2(p)
plt.plot(p,Ent)
plt.title("-p*log2(p) curve graph")
plt.xlabel("p")
plt.ylabel("-p*log2(p)")
plt.show()