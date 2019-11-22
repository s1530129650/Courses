#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: ASM.py
@time: 11/10/2019 4:07 PM
'''
import numpy as np

def IsZero(data):
	if data > 1e-5 or data < -1e-5:
		return False
	return True

# solve Equality Constrained Quadratic Programming
# f(x) = 1/2*d.T*G*d+p.T*d     s.t. A*d = b
#直接应用拉格朗日乘子法
def ECQP(G, p, A, b):
	rowA, colA = A.shape
	left = np.vstack((G, A))#按竖直方向堆砌
	right = np.vstack((A.T, np.zeros((rowA, rowA))))

	x = np.linalg.solve(np.hstack((left, right)), np.vstack((-p, b)))#求解Ax = b

	solution = x[:colA]
	lamb = x[colA:]
	return solution, lamb

# active set method
# CONSTRAINT: Gx <= h Ax = b
def ASM(x0, hessian, p, IE, IEB, A=[[0]], b=0):
    # Step 1: find active set indices
    # Ax = b may be empty
    A = np.array(A)
    zeroIdx = []
    print("type A",type(A))
    if A.shape[0] == len(x0):
        rowA, colA = A.shape
    else:
        rowA = 0
    rowG, colG = IE.shape

    sum = np.dot(IE, x0)
    tmp = sum - IEB
    for i in range(rowG):
        if IsZero(tmp[i]):
            zeroIdx.append(i)
    iter = 0

    x = x0
    while True and iter < 200:
        # Step 2 solve equality constraint optimization
        pNew = hessian.dot(x) + p
        ANew = np.zeros((rowA + len(zeroIdx), colG))
        bNew = np.zeros((rowA + len(zeroIdx), 1))

        for i in range(rowA):
            for j in range(colG):
                ANew[i][j] = A[i][j]
        count = 0
        for i in zeroIdx:
            for j in range(colG):
                ANew[rowA + count][j] = IE[i][j]
            count += 1
        d, lamb = ECQP(hessian, pNew, ANew, bNew)

        is_zero_vector = True
        if np.linalg.norm(d) > 1e-5:
            is_zero_vector = False
        if is_zero_vector:
            # if IsZeroVec(d):
            # for i in range(len(d)):
            #	if  np.linalg.norm(d) > 1e-5:
            #		is_zero_vector = False
            #		break
            # Step 3 if d is a zero vector
            isOpt = True
            idxMin = np.argmin(lamb)
            lambMin = lamb[idxMin]
            if lambMin < 0:
                isOpt = False

            if isOpt:
                output = np.zeros((rowG, 1))
                shift = 0
                for i in zeroIdx:
                    output[i][0] = lamb[shift]
                    shift += 1
                return x, output
                break
            else:
                zeroIdx.remove(idxMin)
            iter += 1
        else:
            # Step 4 if not
            alpha = 1.
            idxMin = -1
            for i in range(rowG):
                if zeroIdx.count(i) == 0:
                    k = (IEB[i][0] - IE[i].T.dot(x)) / (IE[i].T.dot(d))
                    if k <= alpha and IE[i].dot(d) > 0:
                        idxMin = i
                        alpha = k
            x = x + alpha * d
            if idxMin != -1:
                zeroIdx.append(idxMin)
            iter += 1
    return x, lamb

#Gx <= h
x0 = np.array([0,0])
hessian = np.array([[2,-1],[-1,2]])
r = np.array([-3,0])
G = np.array([[1,1],[-1,0],[0,-1]])
h = np.array([2,0,0])
ASM(x0, hessian, r, G, h, A=[[0]], b=0)