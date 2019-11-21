#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: coin.py
@time: 11/20/2019 8:17 PM
'''
import random
import matplotlib.pyplot as plt
random.seed(1)
def biased_coin_flip():
    if random.random() <= 0.6:
        return 1
    else:
        return 0
n_trials = 100
coin_flips = [biased_coin_flip() for _ in range(n_trials)]
n_heads = sum(coin_flips)
print(n_heads)
import pymc3 as pm
with pm.Model() as coin_model:
    p = pm.Normal('x', mu=0.5, sigma=0.1)
    obs = pm.Bernoulli('obs', p, observed=coin_flips)
    step = pm.Metropolis()
    trace = pm.sample(100000, step=step)
    trace = trace[5000:]
p_heads = 0.575
fig, ax = plt.subplots(figsize=(12, 9))
plt.title('Posterior distribution of $p')
plt.vlines(p_heads, 0, n_trials / 10, linestyle='--', label='true $p$ (unknown)')
plt.hist(trace['p'], range=[0.3, 0.9], bins=25, histtype='stepfilled', normed=True)
plt.legend()