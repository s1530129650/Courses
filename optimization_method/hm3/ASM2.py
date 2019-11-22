#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: ASM2.py
@time: 11/10/2019 5:00 PM
'''

# 有效集算法实现，解决二次规划问题
# min 1/2x.THx+c.Tx
# s.t. Ax>=b
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from scipy.stats import ortho_group
from matplotlib import pyplot as plt
def drawContour(data_x,data_y,xstar):
    x_arange = np.linspace(-0.5, 3.0, 256)
    y_arange = np.linspace(-0.5, 3.0, 256)
    X, Y = np.meshgrid(x_arange, y_arange)
    Z1 =  X ** 2 + Y ** 2-X*Y-3*X
    Z2= -X - Y + 2
    Z3 = X
    Z4 = Y
    plt.xlabel('x')
    plt.ylabel('y')
    nm =  3
    plt.contourf(X, Y, Z1, nm, alpha=0.75, cmap='gray_r')
    plt.contourf(X, Y, Z2, nm, alpha=0.75, cmap='gray_r')
    plt.contourf(X, Y, Z3, nm, alpha=0.75, cmap='gray_r')
    plt.contourf(X, Y, Z4, nm, alpha=0.75, cmap='gray_r')
    C1 = plt.contour(X, Y, Z1, nm, colors='green')
    C2 = plt.contour(X, Y, Z2, nm, colors='blue')
    C3 = plt.contour(X, Y, Z3, nm, colors='red')
    C4 = plt.contour(X, Y, Z4, nm, colors='black')
    plt.clabel(C1, inline=1, fontsize=10)
    plt.clabel(C2, inline=1, fontsize=10)
    plt.clabel(C3, inline=1, fontsize=10)
    plt.clabel(C4, inline=1, fontsize=10)
    plt.plot(data_x, data_y, marker = "*",c = "y",ms ="15")
    plt.scatter(xstar[0], xstar[1], marker=(5, 1), c="r", s=1000)
    #plt.plot(1,0,"*", ms = 10)
    plt.show()

class Active_set(object):
    def __init__(self, H, c, A, b):
        self.H = H
        self.c = c
        self.A = A
        self.b = b
        self.epsilon = 1e-6

    def initial_set(self):
        # 选定初始有效集和初始可行解
        # 为了简单只选择第一行
        activae_set_rows = [0]

        # 初始可行解为满足第一个等式的任意解
        # 只需要找到第一个不为0的系数，让其他的变量都为0就可以了
        # np.where返回tuple,第一个元素是索引列表，第二个元素是数据类型，因此需要[0][0]
        idx = activae_set_rows[0]
        index = np.where(self.A[idx] != 0)[0][0]
        value = self.A[idx][index]#系数
        feasible_x = np.zeros(len(self.A[idx]))
        #print("b shape",b.shape)
        feasible_x[index] = self.b[idx][0]/float(value)

        feasible_x = feasible_x.reshape(-1, 1)

        return activae_set_rows, feasible_x

    def calculate_delta(self, x):
        # 计算在x处的导数
        return np.matmul(self.H, x) + self.c

    def find_active_set(self):
        activate_set_rows, feasible_x = self.initial_set()
        steps = 0
        print("steps is {}".format(steps))
        print("initial x is {}".format(feasible_x.flatten()))
        print("Active set is {}".format(activate_set_rows))
        print(30*"*")
        data_x = [feasible_x.flatten()]
        #print("data_x", data_x)
        while True:
            steps += 1
            print("steps is {}".format(steps))

            # 计算在可行解处的导数,作为新的c
            partial_x = self.calculate_delta(feasible_x)
            actual_A, actual_b, actual_b1 = find_new_data(self.A, self.b, activate_set_rows)
            # 利用Lagrange求得等式约束的解
            delta = Lagrange(self.H, partial_x, actual_A, actual_b)
            # print(20*"#")
            # print("delta",delta)
            # input()
            solution = delta[0: self.H.shape[1]]

            if np.sum(np.abs(solution)) < self.epsilon:
                # if np.all(solution == 0):
                # 判断lagrang因子是否全部大于0
                #outcome = Lagrange(self.H, self.c, actual_A, actual_b1)
                outcome = Lagrange(self.H, self.c, actual_A, actual_b1)
                lambda_value = outcome[self.H.shape[1]:].flatten()
                min_value = lambda_value.min()
                if min_value >= 0:
                    print("Reach Optimization")
                    print("Optimize x is {}".format(feasible_x.flatten()))
                    print("Active set is {}".format(activate_set_rows))
                    print("lambda_value is {}".format(lambda_value))
                    print(30*"*")

                    break
                else:
                    index = np.argmin(lambda_value)
                    activate_set_rows.pop(index)
                    # 可行解不变
                    feasible_x = feasible_x
                    print("Not Reach Optimization")
                    print("Feasible x is {}".format(feasible_x.flatten()))
                    print("Active set is {}".format(activate_set_rows))
                    print("lambda_value is {}".format(lambda_value))
                    #print("Min Value is {}".format(0.5*np.matmul(np.matmul(feasible_x.T, self.H), feasible_x)[0][0] + (self.c.T@feasible_x)[0][0]))
                    print(30*"*")
            else:
                in_row, alpha_k = self.calculate_alpha(feasible_x, solution.reshape(-1, 1), activate_set_rows)
                if in_row == -1:
                    alpha = 1
                else:
                    alpha = min(1, alpha_k)

                feasible_x += alpha * solution
                data_x.append(feasible_x.flatten())
                if alpha != 1:
                    activate_set_rows.append(in_row)
                    print("Not Reach Optimization")
                    print("Feasible x is {}".format(feasible_x.flatten()))
                    print("Active set is {}".format(activate_set_rows))
                    print(30*"*")

                    continue
                else:
                    #outcome = Lagrange(self.H, self.c, actual_A, actual_b1)
                    outcome = Lagrange(self.H, self.c, actual_A, actual_b1)
                    lambda_value = outcome[self.H.shape[1]:].flatten()
                    print("min_value",lambda_value)
                    min_value = lambda_value.min()
                    if min_value >= 0:
                        print("Reach Optimization")
                        print("Optimize x is {}".format(feasible_x.flatten()))
                        print("Active set is {}".format(activate_set_rows))
                        print("lambda_value is {}".format(lambda_value))
                        print("Min Value is {}".format(0.5*np.matmul(np.matmul(feasible_x.T, self.H), feasible_x)[0][0] + (self.c.T@feasible_x)[0][0]))
                        print(30*"*")

                        break
                    else:
                        index = np.argmin(lambda_value)
                        activate_set_rows.pop(index)
                        print("Not Reach Optimization")
                        print("Feasible x is {}".format(feasible_x))
                        print("Active set is {}".format(activate_set_rows))
                        print("lambda_value is {}".format(lambda_value))
                        print(30*"*")

            #print("data_x", data_x)
        #print("data_x,before", data_x)
        data_x = np.array(data_x)
        #print("data_x", data_x)
        drawContour(data_x[:, 0], data_x[:, 1], feasible_x)


    def calculate_alpha(self, x, d, activate_set_rows):

        min_alpha = 0
        inrow = -1
        for i in range(self.A.shape[0]):
            if i in activate_set_rows:
                continue
            else:
                b_i = self.b[i][0]
                a_i = self.A[i].reshape(-1, 1)
                #print("a_i  ",a_i,"d   ",d)
                low_number = np.matmul(a_i.T, d)
                if low_number >= 0:
                    continue
                else:
                    new_alpha = (b_i - np.matmul(a_i.T, x)[0][0])/float(low_number)
                    if inrow == -1:
                        inrow = i
                        min_alpha = new_alpha
                    # elif new_alpha < min_alpha and new_alpha != 0:
                    elif new_alpha < min_alpha:
                        min_alpha = new_alpha
                        inrow = i
                    else:
                        continue
        return inrow, min_alpha


def Lagrange(H, c, A, b):
    # 构造lagrange矩阵，然后求逆求解
    up_layer = np.concatenate((H, -A.T), axis=1)
    zero_0 = np.zeros([A.shape[0], A.shape[0]])
    low_layer = np.concatenate((A, zero_0), axis=1)
    lagrange_matrix = np.concatenate((up_layer, low_layer), axis=0)
    #e,v = np.linalg.eig(lagrange_matrix)

    #print("e",e)
    actual_b = np.concatenate((-c, b), axis=0)
    '''
    lagrange_matrix_inverse = np.linalg.inv(lagrange_matrix)
    return np.matmul(lagrange_matrix_inverse, actual_b)
    '''
    return np.linalg.solve(lagrange_matrix.T @ lagrange_matrix, lagrange_matrix.T @actual_b)



def find_new_data(A, b, activate_set_rows):
    # 注意activate_set_rows为空的情形
    actual_A = A[activate_set_rows]
    actual_b = np.zeros_like(b[activate_set_rows])
    return actual_A, actual_b, b[activate_set_rows]

def create_H_c(dimension,matrix_type):
    eigvals = 10*np.abs(np.random.random((dimension)))
    A = np.eye(dimension)
    m = abs(np.random.randint(10))
    #print(m)
    for i in range(dimension):
        A[i][i] = eigvals[i]
    seed = 1
    U= np.float32(ortho_group.rvs(dim=dimension, random_state=seed))
    b = np.random.rand(dimension, 1)
    A = U.transpose().dot(A).dot(U)
    if matrix_type == "convex":
        return A,b
    if matrix_type == "consistently convex":
        return A+m*np.eye(dimension),b,m
    if matrix_type == "bounded convex":
        return

if __name__ == "__main__":
    '''
    H = np.array([[2, 0], [0, 2]])
    c = np.array([-2, -5]).reshape(-1, 1)
    A = np.array([[1, -2], [-1, -2], [-1, 2], [1, 0], [0, 1]])
    b = np.array([-2, -6, -2, 0, 0]).reshape(-1, 1)
   '''
    H = np.array([[2, -1], [-1, 2]])
    c = np.array([-3, 0]).reshape(-1, 1)
    A = np.array([[-1, -1], [1, 0], [0, 1]])
    b = np.array([-2, 0, 0]).reshape(-1, 1)

    H = np.array([[2, -1], [-1, 2]])
    c = np.array([-3, 0]).reshape(-1, 1)
    A = np.array([[1, 0],[-1, -1],  [0, 1]])
    b = np.array([0, -2, 0]).reshape(-1, 1)

    '''
    H = np.array([[2, 0], [0, 2]])
    c = np.array([-2, -5]).reshape(-1, 1)
    A = np.array([[1, -2], [-1, -2], [-1, -2], [1, 0], [0, 1]])
    b = np.array([-2, -6, -2, 0, 0]).reshape(-1, 1)

    dimension = 100
    matrix_type = "convex"
    np.random.seed(0)
    H,c = create_H_c(dimension,matrix_type)
    c = c.reshape(-1,1)
    A = np.random.randint(0,10,size = [5,dimension])
    b= np.random.randint(0,10,size = [dimension,1]).reshape(-1, 1)
   
    H = np.array([[2, 0], [0, 2]])
    c = np.array([-1, -1]).reshape(-1, 1)
    A = np.array([[-1, -1], [1, 0], [0, 1]])
    b = np.array([-2, 0, 0]).reshape(-1, 1)
'''
    test = Active_set(H, c, A, b)

    # if A.shape[1] < 3:
    #     for i in range(3):
    #
    #         plt.plot([b[i]/A[i][0],0] , [0,b[i]/A[i][1]] , color='r')

    plt.show()
    test.find_active_set()


