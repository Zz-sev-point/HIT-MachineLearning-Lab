import numpy as np
import sys

class GradientDescent(object):
    def __init__(self, X, Y, w, m, n, lamda, step, epsilon):
        self.X = X
        self.Y = Y
        self.w = w
        self.m = m
        self.n = n
        self.lamda = lamda
        self.step = step
        self.epsilon = epsilon

    def sigmoid_func(self, x):
        return 1/(1+np.exp(-x))

    def likelihood_func(self, w):
        amount = np.size(self.X, axis=0)
        p = np.zeros((amount, 1))
        sum = 0
        for i in range(amount):
            p[i] = np.dot(w, self.X[i].T)
            # 当p[i]足够大时，进行近似处理防止溢出
            if(p[i] >= np.log(sys.float_info.max/2)):
                sum += p[i]
            else:
                sum += np.log(1+np.exp(p[i]))
        return np.dot(self.Y, p) - sum
    
    def partial_derivative(self, w):
        return np.dot(self.sigmoid_func(np.dot(w, self.X.T))-self.Y, self.X)

    # 梯度下降法,m为样本数,n为参数个数，lamda为惩罚项系数
    # step为步长，epsilon为迭代误差，dimension为X样本维度
    def gradient_descent(self):
        w = self.w
        # 记录损失函数值的变化情况
        losslist = []
        counterlist = []
        i = 1
        j = 0
        while 1:
            OldLoss = -self.likelihood_func(w)/self.m
            gradient = self.partial_derivative(w)/self.m
            losslist.append(OldLoss)
            counterlist.append(i)
            w = w - self.step*self.lamda*w - self.step*gradient
            NewLoss = -self.likelihood_func(w)/self.m
            i = i+1
            j = j+1
            # gnorm = np.dot(gradient, gradient.T)
            # print(OldLoss-NewLoss)
            # print(OldLoss)
            # 若损失函数收敛则结束循环
            if abs(OldLoss-NewLoss) < self.epsilon:
                losslist.append(NewLoss)
                counterlist.append(i)
                break
            else:
                if OldLoss < NewLoss:
                    self.step *= 0.5
                    j = 0
                if j>10000:
                    self.step *= 2
                    j = 0
        return w, losslist, counterlist
