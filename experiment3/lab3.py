import numpy as np
import matplotlib.pyplot as plt
import k_means
import em_gmm
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
from math import log
from scipy.stats import multivariate_normal
import itertools


# sample_means为各类分布的样本均值，sample_amount为各类分布的样本量，k为k类Gaussion Distribution
def generate_data(sample_means, sample_amount, k):
    cov = [[0.5, 0.01], [0.01, 0.5]]
    data = []
    for i in range(k):
        for j in range(sample_amount[i]):
            data.append(np.random.multivariate_normal(
                (sample_means[i][0], sample_means[i][1]), cov))
    return np.array(data)

def loss_show(loss, counter):
    fig = plt.figure(1)
    plt.title("Likelhood")
    plt.plot(counter, loss, '-')
    plt.show()

# load UCI dataset
def load_iris():
    load_data = pd.read_csv('./iris.csv')
    data = np.array(load_data.drop('class', axis=1))
    labels = load_data['class']
    k = 3
    return data, labels, k

# GMM
k = 3
means = [[3, 3], [0, -4], [-1, 2]]
amount = [50, 100, 150]
data = generate_data(means, amount, k)

# k = 5
# means = [[3, 3], [2, -4], [-2, 3], [-1, -1], [0, 3]]
# amount = [30, 30, 40, 35, 40]
# data = generate_data(means, amount, k)

# k_means
# km = k_means.K_Means(data, k, 1e-15)
# km.classify()
# km.show()

# EM
# eg = em_gmm.EM_GMM(data, k, 1e-15)
# loss, counter = eg.classify()
# eg.show()
# loss_show(loss, counter)
# print("means: ",eg.mu)
# print("cov: ",eg.sigma)


# UCI experiment
l = list(itertools.permutations(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 3))
data, labels, k = load_iris()
iris_km = k_means.K_Means(data, k, 1e-15)
iris_km.classify()
# iris_eg = em_gmm.EM_GMM(data, k, 1e-15)
# iris_eg.classify()

# calculate accuracy
acc_km = []
acc_eg = []
for i in range(len(l)):
    count_km = 0
    count_eg = 0
    for j in range(iris_km.n):
        if labels[j] == l[i][int(iris_km.label[j])]:
            count_km += 1
        # if labels[j] == l[i][int(iris_eg.labels[j])]:
        #     count_eg += 1
    acc_km.append(count_km/iris_km.n)
    # acc_eg.append(count_eg/iris_eg.n)
print('UCI Iris')
print('K-Means accuracy: ')
print(np.max(acc_km))
# print('EM accuracy: ')
# print(np.max(acc_eg))
