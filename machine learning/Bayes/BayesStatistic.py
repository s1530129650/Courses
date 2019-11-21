#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: BayesStatistic.py
@time: 11/16/2019 9:19 PM
'''

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
n = 10
D = 2*np.random.randn(10)+1
with pm.Model() as coin_model:
    p = pm.Normal('p', mu= 1, sigma= 2 )
    obs = pm.Normal('obs', p, observed=D)
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step,chains=1)
    trace = trace[500:]
p_heads = 0.575
fig, ax = plt.subplots(figsize=(12, 9))
plt.title('Posterior distribution of $p')
#plt.vlines(p_heads, 0, n_trials / 10, linestyle='--', label='true $p$ (unknown)')
plt.hist(trace['p'], range=[0.3, 0.9], bins=25, histtype='stepfilled', normed=True)
plt.legend()

