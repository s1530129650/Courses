#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: Beta.py
@time: 11/21/2019 3:30 PM
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.linspace(0, 1, 200)
alphas = [5, 5., 100., 1000]
betas = [5, 6., 109., 956.]
for a, b in zip(alphas, betas):
    pdf = st.beta.pdf(x, a, b)
    plt.plot(x, pdf, label=r'$\beta_z$ = {}, $\beta_B$ = {}'.format(a, b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
#plt.ylim(0, 10)
plt.xticks(np.arange(0, 1, 0.1))
plt.legend(loc=9)
plt.show()