import numpy as np

class Newton(object):
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
    
    def partial_derivative(self, w):
        return np.dot(self.sigmoid_func(np.dot(w, self.X.T))-self.Y, self.X) + self.lamda*w

    def second_derivative(self, w):
        ans = np.eye(self.n) * self.lamda
        for i in range(self.m):
            temp = self.sigmoid_func(np.dot(w, self.X[i].T))
            ans += self.X[i] * np.transpose([self.X[i]]) * temp * (1 - temp)
        return ans

    def newton(self):
        w = self.w
        while 1:
            gradient = self.partial_derivative(w)
            gnorm = np.linalg.norm(gradient)
            print(gnorm)
            if gnorm < self.epsilon:
                break
            w = w - np.dot(gradient, np.linalg.pinv(self.second_derivative(w)))
        return w