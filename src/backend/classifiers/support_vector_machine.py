import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cvxopt import matrix, solvers


class SupportVectorMachine(object):
    def __init__(self, C=1, kernel='linear', power=3, gamma=None):
        self.C = C
        self.kernel_name = kernel
        self.power = power
        self.gamma = gamma
        self.number_of_samples = None
        self.alphas = None
        self.b = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.number_of_samples, number_of_features = np.shape(X)

        # Set gamma to 1/n_features by default and this is required only in RBF kernel
        if not self.gamma:
            self.gamma = 1 / number_of_features

        # Calculation of P
        P = np.zeros((self.number_of_samples, self.number_of_samples))
        for i in range(self.number_of_samples):
            for j in range(self.number_of_samples):
                P[i][j] = y[i] * y[j] * self.kernels(self.kernel_name, X[i], X[j])
        # Converting numpy into convex optimization matrix
        P = matrix(P)
        q = matrix(np.ones(self.number_of_samples) * -1)
        A = matrix(y, (1, self.number_of_samples), tc='d')
        b = matrix(0, tc='d')

        if not self.C:
            G = matrix(np.identity(self.number_of_samples) * -1)
            h = matrix(np.zeros(self.number_of_samples))
        else:
            G_max = np.identity(self.number_of_samples) * -1
            G_min = np.identity(self.number_of_samples)
            G = matrix(np.vstack((G_max, G_min)))
            h_max = matrix(np.zeros(self.number_of_samples))
            h_min = matrix(np.ones(self.number_of_samples) * self.C)
            h = matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        solution = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        self.alphas = np.array(solution['x']).reshape(self.number_of_samples)
        # Extract support vector indices
        support_vectors = np.where(self.alphas > 1e-4)[0][0]
        self.b = y[support_vectors] - sum(self.alphas * y * self.kernels(self.kernel_name, X, X[support_vectors]))

    def predict(self, u):
        y_pred = []
        for sample in u:
            if self.b + sum(self.alphas * self.y * self.kernels(self.kernel_name, self.X, sample)) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        print(y_pred)
        return np.array(y_pred)

    def kernels(self, name, x, y, gamma=0):
        if name == 'linear':
            return np.dot(x, y.T)
        if name == 'poly':
            return (1 + np.dot(x, y.T)) ** 3
        if name == 'rbf':
            distance = np.linalg.norm(x - y) ** 2
            return np.exp(-gamma * distance)

    def plot(self):
        x_min = min(self.X[:, 0]) - 0.5
        x_max = max(self.X[:, 0]) + 0.5
        y_min = min(self.X[:, 1]) - 0.5
        y_max = max(self.X[:, 1]) + 0.5
        step = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        d = np.concatenate((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)), axis=1)
        Z = self.b + np.sum(self.alphas.reshape((self.number_of_samples, 1)) * self.y.reshape(
            (self.number_of_samples, 1)) * self.kernels(self.kernel_name, self.X, d), axis=0)
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots()
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, ax=ax)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1])
        plt.show()