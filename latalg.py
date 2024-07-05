import numpy as np

class lattice():
    def __init__(self, b, n, m):
        self.basis = b[:]
        self.nrows = n
        self.ncols = m
        self.mu = np.zeros((n, n))
        self.basis_star = np.zeros((n, m))
        self.B = np.zeros(n)


    def GSO(self, mode = "normal"):
        for i in range(self.nrows):
            self.basis_star[i] = self.basis[i, :]
            for j in range(i):
                self.mu[i, j] = (self.basis[i] @ self.basis_star[j]) / (self.basis_star[j] @ self.basis_star[j])
                self.basis_star[i] -= self.mu[i, j] * self.basis_star[j, :]
            if mode == "square" or mode == "both":
                self.B[i] = self.basis_star[i] @ self.basis_star[i]
       
        if mode == "normal":
            return self.basis_star, self.mu
        elif mode == "square":
            return self.B, self.mu
        elif mode == "both":
            return self.basis_star, self.B, self.mu
       

    def SizeReduce(self, i, j):
        if abs(self.mu[i, j]) > 0.5:
            q = round(self.mu[i, j])
            self.b[i] -q * self.b[j, :]
            self.mu[i, : j + 1] -= q * self.mu[j, : j + 1]
        return self.b, self.mu