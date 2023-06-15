ge(n):
        if data_label[i, dim] == 0:
            ax.scatter(data_label[i, 0], data_label[i, 1], alpha=0.7, c='blue')
        elif data_label[i, dim] == 1:
            ax.scatter(data_label[i, 0], data_label[i, 1],
                       alpha=0.7, c='green')
        elif data_label[i, dim] == 2:
            ax.scatter(data_label[i, 0], data_label[i, 1], alpha=0.7, c='red')