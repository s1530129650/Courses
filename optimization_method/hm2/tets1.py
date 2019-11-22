#!/usr/bin/python
# coding: utf-8

# 面向对象的方式(没有交互的效果)

import numpy as np
import matplotlib.pyplot as plt


def bfgs(fun, gfun, x0, threshold=10 ** -6):
    result = []
    gradient = []
    maxk = 500
    rho = 0.55
    sigma = 0.4
    m = shape(x0)[0]
    Bk = eye(m)
    k = 0
    while (k < maxk):
        gk = mat(gfun(x0))  # 计算梯度
        dk = mat(-linalg.solve(Bk, gk))
        m = 0
        mk = 0
        while (m < 20):
            newf = fun(x0 + rho ** m * dk)
            oldf = fun(x0)
            if (newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0, 0]):
                mk = m
                break
            m = m + 1

        # BFGS校正
        x = x0 + rho ** mk * dk
        sk = x - x0
        yk = gfun(x) - gk
        if (yk.T * sk > 0):
            Bk = Bk - (Bk * sk * sk.T * Bk) / (sk.T * Bk * sk) + (yk * yk.T) / (yk.T * sk)

        k = k + 1
        x0 = x
        if np.linalg.norm(gfun(x)) < threshold:
            break
        gradient.append(np.linalg.norm(gfun(x)))
        result.append(mat(fun(x0)).A)

    return x, fun(x), result, gradient