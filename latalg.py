import numpy as np

class lattice():
    def __init__(self, b: np.ndarray, n: int, m: int):
        self.basis = b[:]
        self.nrows = n
        self.ncols = m
        self.mu = np.zeros((n, n))
        self.basis_star = np.zeros((n, m))
        self.B = np.zeros(n)
    
    def __init__(self, b: np.ndarray):
        n, m = b.shape
        self.basis = b[:]
        self.nrows = n
        self.ncols = m
        self.mu = np.eye(n)
        self.basis_star = np.zeros((n, m))
        self.B = np.zeros(n)


    def random_lattice(self, n: int):
        self.basis = np.random.randint(1000)
        self.nrows = n
        self.ncols = n
        self.mu = np.eye(n)
        self.basis_star = np.zeros((n, n))
        self.B = np.zeros(n)


    def GSO(self, mode: str = "normal") -> np.ndarray:
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
       

    def SizeReduce(self, i: int, j: int):
        if abs(self.mu[i, j]) > 0.5:
            q = round(self.mu[i, j])
            self.basis[i] -= q * self.basis[j, :]
            self.mu[i, : j + 1] -= q * self.mu[j, : j + 1]
        return self.basis, self.mu
    

    def LLL(self, delta: float = 0.99):
        self.B, self.mu = self.GSO(mode = "square")

        k = 1
        while k < self.nrows:
            # size-reduction
            for j in range(k)[::-1]: self.basis, mu = self.SizeReduce(k, j)

            # Satisfies Lovasz condition or not
            if self.B[k] >= (delta - self.mu[k, k - 1] * self.mu[k, k - 1]) * self.B[k - 1]:
                k += 1
            else:
                self.basis[k], self.basis[k - 1] = self.basis[k - 1, :], self.basis[k, :]

                # Updates GSO-information
                nu = self.mu[k, k - 1]
                b = self.B[k] + nu * nu * self.B[k - 1]; b_inv = 1. / b
                self.mu[k, k - 1] = nu * self.B[k - 1] * b_inv
                self.B[k] *= self.B[k - 1] * b_inv
                self.B[k - 1] = b
                tmp = self.mu[k - 1][: k - 1]
                self.mu[k - 1][: k - 1] = self.mu[k][: k - 1]
                self.mu[k][: k - 1] = tmp[:]
                t = mu[k + 1 : self.nrows, k]
                mu[k + 1 : self.nrows, k] = mu[k + 1 : self.nrows, k - 1] - nu * t[:]
                mu[k + 1 : self.nrows, k - 1] = t + mu[k, k - 1] * mu[k + 1 : self.nrows, k]

                k = max(k - 1, 1)
        return self.basis
    

    def DeepLLL(self, delta : float = 0.99):
        self.B, self.mu = self.GSO(mode = "square")
        k : int = 1
        
        while k < self.nrows:
            for j in range(k)[::-1]: self.basis, self.mu = self.SizeReduce(k, j)
            C = np.linalg.norm(self.basis[k])
            i = 0
            while i < k:
                if C >= delta * self.B[i]:
                    C -= self.mu[k, i] * self.mu[k, i] * self.B[i]
                    i += 1
                else:
                    v = self.basis[k, :]
                    self.basis[i + 1: k + 1] = self.basis[i: k]
                    self.basis[i] = v[:]
                    self.B, self.mu = self.GSO(mode = "square")
                    k = max(i - 1, 0)
            k += 1
        
        return self.basis