import numpy as np

class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centerX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centerX = self.centralized()
        self.C = self.cov()
        self.U = self._U()
        self.Z = self._Z()

    def centralized(self):
        '''中心化'''
        centerX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centerX = self.X - mean
        return centerX

    def cov(self):
        '''求协方差矩阵'''
        ns = np.shape(self.centerX)[0]
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        return C

    def _U(self):
        '''求降维转换矩阵'''
        a, b = np.linalg.eig(self.C)
        idx = np.argsort(-1*a)
        UT = [b[:, idx[i]] for i in range(self.K)]
        print('UT.shape:', np.shape(UT))
        U = np.transpose(UT)
        print('U.shape:', np.shape(U))
        return U

    def _Z(self):
        '''求降维后的矩阵'''
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        Z = np.dot(self.X, self.U)
        print('Z shape:', np.shape(Z))
        return Z


if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = 2
    pca = CPCA(X, K)