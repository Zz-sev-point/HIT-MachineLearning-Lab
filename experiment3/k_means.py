import numpy as np
import matplotlib.pyplot as plt

color = ['blue', 'red', 'green', 'yellow', 'cyan']
center_color = ['black', 'brown', 'gold', 'pink', 'purple']

class K_Means(object):
    def __init__(self, data, k, e):
        self.data = data
        self.k = k
        self.n = np.size(self.data, axis=0)
        self.dim = np.size(self.data, axis=1)
        self.e = e
        self.label = np.zeros(self.n)
        self.centers = np.zeros((self.k, self.dim))

    def euclidean_distance(self, a, b):
        return np.linalg.norm(a-b)

    def center_confirm(self):
        self.n = np.size(self.data, axis=0)
        index = np.random.randint(0,self.n)
        cen = [self.data[index]]
        for i in range(self.k-1):
            dis = []
            for j in range(self.n):
                dis.append(np.sum(self.euclidean_distance(self.data[j], cen[self.k]) for self.k in range(len(cen))))
            cen.append(self.data[np.argmax(dis)])
        return np.array(cen)

    def show(self):
        # scatter
        fig = plt.figure()
        plt.title('K_Means')
        # ax = fig.add_subplot(111)
        for i in range(self.n):
            plt.scatter(self.data[i, 0], self.data[i, 1], alpha=0.7, c=color[int(self.label[i])])
        # scatter centers
        for i in range(self.k):
            plt.scatter(self.centers[i,0], self.centers[i,1], alpha=0.7, c=center_color[i])
        plt.show()

    def classify(self):
        # not random
        # self.centers = self.center_confirm()
        # random
        for i in range(self.k):
            self.centers[i, :] = self.data[np.random.randint(low=0, high=self.n), :]
        # iterate
        while 1:
            distance = np.zeros(self.k)
            amou = np.zeros(self.k)
            new_centers = np.zeros((self.k, self.dim))
            # update labels
            for i in range(self.n):
                for j in range(self.k):
                    distance[j] = np.linalg.norm(self.data[i, :] - self.centers[j, :])
                new_label = np.argmin(distance)
                self.label[i] = new_label

            converge_counter = 0
            sum_bias = 0
            # update centers
            for i in range(self.n):
                label = int(self.label[i])
                new_centers[label] += self.data[i]
                amou[label] += 1
            for i in range(self.k):
                if amou[i] != 0:
                    new_centers[i] /= amou[i]
                bias = np.linalg.norm(new_centers[i]-self.centers[i])
                sum_bias += bias
                if bias < self.e:
                    converge_counter += 1
            print(sum_bias)
            if converge_counter == self.k:
                break
            else:
                self.centers = new_centers
        return self.label, self.centers