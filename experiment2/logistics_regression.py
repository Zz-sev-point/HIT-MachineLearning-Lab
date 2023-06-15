import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradient_descent
import newton
from mpl_toolkits.mplot3d import Axes3D


def get_data(SampleAmount, naive):
    boundary = np.ceil(SampleAmount/2).astype(np.int32)
    lam = 0.2
    cov = 0.1
    X_mean0 = [-0.6, -0.6]
    X_mean1 = [0.6, 0.6]
    X = np.zeros((SampleAmount, 2))
    train_X = np.ones((SampleAmount, 3))
    Y = np.zeros(SampleAmount)
    # 满足朴素贝叶斯
    if naive:
        X[:boundary, :] = np.random.multivariate_normal(
            X_mean0, [[lam, 0], [0, lam]], size=boundary)
        X[boundary:, :] = np.random.multivariate_normal(
            X_mean1, [[lam, 0], [0, lam]], size=SampleAmount-boundary)
        Y[:boundary] = 0
        Y[boundary:] = 1
    # 不满足朴素贝叶斯
    else:
        X[:boundary, :] = np.random.multivariate_normal(
            X_mean0, [[lam, cov], [cov, lam]], size=boundary)
        X[boundary:, :] = np.random.multivariate_normal(
            X_mean1, [[lam, cov], [cov, lam]], size=SampleAmount-boundary)
        Y[:boundary] = 0
        Y[boundary:] = 1
    train_X[:, 1] = X[:, 0]
    train_X[:, 2] = X[:, 1]
    return X, Y, train_X

# 画二维图
def graph(X, Y, w):
    plt.scatter(X[:, 0], X[:, 1], c=Y, label="sample")
    dimension = np.size(w, axis=1)
    w = w.reshape(dimension)
    coeff = -(w/w[dimension-1])[0:dimension-1]
    decisionboundary = np.poly1d(coeff[::-1])
    result_Y = decisionboundary(X[:, 0])
    plt.plot(X[:, 0], result_Y, linestyle='-', color='k',
             marker='', label="decision boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Logistic Regression")
    plt.legend(loc="upper right")
    plt.show()
    return 0

# 画三维图
def graph_3D(X, Y, w):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    dimension = np.size(w, axis=1)
    w = w.reshape(dimension)
    coeff = -(w/w[dimension-1])[0:dimension-1]
    x = np.arange(np.min(X[:, 0]), np.max(X[:, 0])+1, 1)
    y = np.arange(np.min(X[:, 1]), np.max(X[:, 1])+1, 1)
    xp, yp = np.meshgrid(x, y)
    z = coeff[0] + coeff[1]*xp + coeff[2]*yp
    ax.plot_surface(x, y, z)
    plt.show()
    return 0

# 画损失函数图像
def graph_loss(loss, counter):
    plt.plot(counter, loss)
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("The Losses of Gradient Descent")
    plt.show()
    return 0

# 计算准确性
def cal_accuracy(test_x, test_y, test_size, dimension, w):
    label = np.ones(test_size)
    correct_count = 0
    xt = np.ones((test_size, dimension+1))
    for i in range(dimension):
        xt[:, i+1] = test_x[:, i]
    for i in range(test_size):
        if np.dot(w, xt[i].T) >= 0:
            label[i] = 1
        else:
            label[i] = 0
        if label[i] == test_y[i]:
            correct_count += 1
    correct_rate = correct_count / test_size
    print("accuracy: ", correct_rate)
    return correct_rate

# 模拟实验
def exp(SampleAmount, w, lamda, step, epsilon, naive):
    X, Y, train_X = get_data(SampleAmount, naive)
    gd = gradient_descent.GradientDescent(
        train_X, Y, w, SampleAmount, np.size(train_X, axis=1), lamda, step, epsilon)
    w1, loss, counter = gd.gradient_descent()
    nt = newton.Newton(train_X, Y, w, SampleAmount, np.size(train_X, axis=1), lamda, step, epsilon)
    w2 = nt.newton()
    graph(X, Y, w1)
    graph_loss(loss, counter)
    graph(X, Y, w2)
    return 0

# 处理UCI数据
def generate_UCI_data(train_rate, step, load_data):
    np.random.shuffle(load_data)  # 打乱数据集以便选出训练集
    load_data_size = np.size(load_data, axis=0)
    train_data = load_data[:int(load_data_size*train_rate), :]
    test_data = load_data[int(load_data_size*train_rate):, :]
    dimension = np.size(load_data, axis=1) - 1
    # 训练集
    train_x = train_data[:, 0:dimension]
    train_x = train_x[::step]
    train_size = np.size(train_x, axis=0)
    train_y = train_data[:, dimension:dimension+1]
    train_y = train_y[::step]
    train_y = train_y.reshape(train_size)
    # 测试集
    test_size = np.size(test_data, axis=0)
    test_x = test_data[:, 0:dimension]
    test_y = test_data[:, dimension:dimension+1].reshape(test_size)
    return train_x, train_y, train_size, test_x, test_y, test_size, dimension

# skin_nonskin experiment
def skin_exp(w, lamda, step, epsilon):
    load_data = np.loadtxt("./Skin_NonSkin.txt", dtype=np.int32)
    load_data[:, 3] = load_data[:, 3] - 1
    x, train_y, train_size, test_x, test_y, test_size, dimension = generate_UCI_data(
        0.5, 30, load_data)
    train_x = np.ones((train_size, dimension+1))
    for i in range(dimension):
        train_x[:, i+1] = x[:, i]
    gd = gradient_descent.GradientDescent(
        train_x, train_y, w, train_size, dimension+1, lamda, step, epsilon)
    w1, loss, counter = gd.gradient_descent()
    graph_3D(x, train_y, w1)
    graph_loss(loss, counter)
    nt = newton.Newton(
        train_x, train_y, w, train_size, dimension+1, lamda, step, epsilon)
    w2 = nt.newton()
    graph_3D(x, train_y, w2)
    # 计算准确率
    print("skin_gradientdescent: ")
    cal_accuracy(test_x, test_y, test_size, dimension, w1)
    print("skin_newton: ")
    cal_accuracy(test_x, test_y, test_size, dimension, w2)
    return 0

# banknote experiment
def banknote_exp(w, lamda, step, epsilon):
    load_data = pd.read_csv("./data_banknote_authentication.csv")
    data = np.array(load_data)
    x, train_y, train_size, test_x, test_y, test_size, dimension = generate_UCI_data(
        0.5, 1, data)
    train_x = np.ones((train_size, dimension+1))
    for i in range(dimension):
        train_x[:, i+1] = x[:, i]
    gd = gradient_descent.GradientDescent(
        train_x, train_y, w, train_size, dimension+1, lamda, step, epsilon)
    w1, loss, counter = gd.gradient_descent()
    graph_loss(loss, counter)
    nt = newton.Newton(
        train_x, train_y, w, train_size, dimension+1, lamda, step, epsilon)
    w2 = nt.newton()
    # 计算准确率
    print("banknote_gradientdescent: ")
    cal_accuracy(test_x, test_y, test_size, dimension, w1)
    print("banknote_newton: ")
    cal_accuracy(test_x, test_y, test_size, dimension, w2)
    return 0

SampleAmount = 300
lamda = 0.0001
step = 0.1
naive = False
w1 = np.zeros((1, 3))
w2 = np.zeros((1, 4))
w3 = np.ones((1, 5))*0
exp(SampleAmount, w1, lamda, step, 0.0001, True)
exp(SampleAmount, w1, lamda, step, 0.0001, False)
skin_exp(w2, lamda, step, 0.00001)
banknote_exp(w3, lamda, step, 0.00001)
