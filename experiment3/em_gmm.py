import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.stats import multivariate_normal

color = ['blue', 'red', 'green', 'yellow', 'cyan']
center_color = ['black', 'brown', 'gold', 'pink', 'purple']


class EM_GMM():
    def __init__(self, data, k, e):
        self.data = data
        self.k = k
        self.e = e
        self.n = np.size(data, axis=0)
        self.dim = np.size(data, axis=1)
        self.labels = np.zeros(self.n)
        self.mu = np.ones((k, self.dim))
        self.sigma = np.ones((k, self.dim, self.dim))
        self.py = np.array([1/k]*k)
        self.gamma = np.zeros((self.n, self.k))
        for i in range(k):
            self.sigma[i] = np.eye(self.dim)
        for i in range(k):
            self.mu[i] *= i+3

    def e_step(self):
        gamma = np.zeros((self.n, self.k))
        for i in range(self.n):
            for j in range(self.k):
                gamma[i][j] = self.py[j] * \
                    multivariate_normal.pdf(
                        self.data[i], self.mu[j], self.sigma[j])
            margin_pro = np.sum(gamma[i])
            gamma[i] /= margin_pro
        self.gamma = gamma

    def m_step(self):
        for i in range(self.k):
            gamma_i = np.expand_dims(self.gamma[:, i], axis=1)
            gamma_sum = np.sum(gamma_i)
            self.py[i] = gamma_sum/self.n
            self.mu[i] = (self.data*gamma_i).sum(axis=0)/gamma_sum
            self.sigma[i] = np.dot(
                (self.data-self.mu[i]).T, (self.data-self.mu[i])*gamma_i)/gamma_sum

    def likelihood(self):
        re = 0
        for i in range(self.n):
            temp = 0
            for j in range(self.k):
                temp += self.py[j]*multivariate_normal.pdf(
                    self.data[i], mean=self.mu[j], cov=self.sigma[j])
            re += log(temp)
        return re

    def show(self):
        # scatter
        fig = plt.figure()
        plt.title('EM')
        # ax = fig.add_subplot(111)
        for i in range(self.n):
            plt.scatter(self.data[i, 0], self.data[i, 1],
                       alpha=0.7, c=color[int(self.labels[i])])
        plt.show()

    def classify(self):
        loss = []
        counter = []
        i = 0
        while 1:
            old_loss = self.likelihood()
            self.e_step()
            self.m_step()
            loss.append(abs(old_loss))
            counter.append(i)
            i += 1
            new_loss = self.likelihood()
            print(old_loss)
            # print(abs(new_loss - old_loss))
            if abs(new_loss - old_loss) < self.e:
                break
        self.labels = np.argmax(self.gamma, axis=1)
        return loss, counter
