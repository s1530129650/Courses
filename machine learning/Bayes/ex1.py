#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: ex1.py
@time: 11/20/2019 1:55 PM
'''
# Imports
import pymc3 as pm
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import seaborn as sns
import missingno as msno

# Set plotting style
# plt.style.use('fivethirtyeight')
sns.set_style('white')
sns.set_context('poster')


import warnings
warnings.filterwarnings('ignore')
# Make the data needed for the problem.
from random import shuffle
total = 30
n_heads = 11
n_tails = total - n_heads
tosses = [1] * n_heads + [0] * n_tails
shuffle(tosses)
# Context manager syntax. `coin_model` is **just**
# a placeholder
with pm.Model() as coin_model:
    # Distributions are PyMC3 objects.
    # Specify prior using Uniform object.
    p_prior = pm.Uniform('p', 0, 1)

    # Specify likelihood using Bernoulli object.
    like = pm.Bernoulli('likelihood', p=p_prior,
                        observed=tosses)
    # "observed=data" is key
    # for likelihood.

with coin_model:
    # don't worry about this:
    step = pm.Metropolis()

    # focus on this, the Inference Button:
    coin_trace = pm.sample(5000, step=step)


pm.traceplot(coin_trace)
plt.show()
