import numpy as np
import time

class lattice():
    def __init__(self, b: np.ndarray, n: int, m: int):
        self.basis = np.copy(b)
        self.nrows = n
        self.ncols = m
        self.mu = np.zeros((n, n))
        self.basis_star = np.zeros((n, m))
        self.B = np.zeros(n)
    
    def __init__(self, b: np.ndarray):
        n, m = b.shape
        self.basis = np.copy(b)
        self.nrows = n
        self.ncols = m
        self.mu = np.eye(n)
        self.basis_star = np.zeros((n, m))
        self.B = np.zeros(n)

    def lattice(self):
        return self.basis

    def print(self):
        print(self.basis)

    def vol(self):
        return np.sqrt(np.linalg.det(self.basis * self.basis.T))

    def GSO(self, mode: str = "normal") -> np.ndarray:
        """
        # Computes Gram-Schmidt orthogonal basis matrix.

        ## Parameters
        b : numpy.ndarray
            A lattice basis matrix(Each basis is row vector).

        ## Returns
        GSOb : numpy.ndarray
            Gram-Schmidt orthogonal basis matrix of an input basis.
        mu : int
            Gram-Schmidt coefficient matrix
        """
        for i in range(self.nrows):
            self.basis_star[i] = np.copy(self.basis[i])
            for j in range(i):
                self.mu[i, j] = (self.basis[i] @ self.basis_star[j]) / (self.basis_star[j] @ self.basis_star[j])
                self.basis_star[i] -= self.mu[i, j] * np.copy(self.basis_star[j])
            if mode == "square" or mode == "both":
                self.B[i] = self.basis_star[i] @ self.basis_star[i]
       
        if mode == "normal":
            return self.basis_star, self.mu
        elif mode == "square":
            return self.B, self.mu
        elif mode == "both":
            return self.basis_star, self.B, self.mu
    

    def LLL(self, delta: float = 0.99) -> np.ndarray:
        """
        LLL-reduces.
        This algorithm is from A. K. Lenstra, H. W. Lenstra, and L. Lovasz (1982)

        Parameters
        ----------
        b : numpy.ndarray
            A lattice basis matrix(Each basis is row vector).
        delta : float
            A reduction parameter(0.5 <= delta <= 1).

        Returns
        -------
        b : numpy.ndarray
            LLL-reduced basis matrix.
        """
        self.B, self.mu = self.GSO(mode = "square")

        k = 1
        while k < self.nrows:
            for j in range(k)[::-1]:
                if abs(self.mu[k, j]) > 0.5:
                    q = round(self.mu[k, j])
                    self.basis[k] -= q * np.copy(self.basis[j])
                    self.mu[k, : j + 1] -= q * np.copy(self.mu[j, : j + 1])

            if self.B[k] >= (delta - self.mu[k, k - 1] * self.mu[k, k - 1]) * self.B[k - 1]:
                k += 1
            else:
                self.basis[k], self.basis[k - 1] = np.copy(self.basis[k - 1, :]), np.copy(self.basis[k])

                # Updates GSO-information
                nu = self.mu[k, k - 1]
                b = self.B[k] + nu * nu * self.B[k - 1]; b_inv = 1. / b
                self.mu[k, k - 1] = nu * self.B[k - 1] * b_inv
                self.B[k] *= self.B[k - 1] * b_inv
                self.B[k - 1] = b
                tmp = np.copy(self.mu[k - 1, : k - 1])
                self.mu[k - 1, : k - 1] = np.copy(self.mu[k, : k - 1])
                self.mu[k][: k - 1] = np.copy(tmp)
                t = np.copy(self.mu[k + 1 : self.nrows, k])
                self.mu[k + 1 : self.nrows, k] = np.copy(self.mu[k + 1 : self.nrows, k - 1]) - nu * np.copy(t)
                self.mu[k + 1 : self.nrows, k - 1] = np.copy(t) + self.mu[k, k - 1] * np.copy(self.mu[k + 1 : self.nrows, k])

                k = max(k - 1, 1)
        return self
    

    def DeepLLL(self, delta : float = 0.99, gamma = 1):
        self.B, self.mu = self.GSO(mode = "square")
        k = 1
        while k < self.nrows:
            for j in range(k)[::-1]:
                if abs(self.mu[k, j]) > 0.5:
                    q = round(self.mu[k, j])
                    self.basis[k] -= q * np.copy(self.basis[j])
                    self.mu[k, : j + 1] -= q * np.copy(self.mu[j, : j + 1])
            C = self.B[k]
            i = 0
            while i < k:
                if C >= delta * self.B[i]:
                    C -= self.mu[k, i] * self.mu[k, i] * self.B[i]
                    i += 1
                else:
                    if gamma > 0 and (0 <= i <= gamma or k - i <= gamma):
                        v = np.copy(self.basis[k])
                        for j in range(i + 1, k + 1)[::-1]:
                            self.basis[j] = np.copy(self.basis[j - 1])
                        #self.basis[i + 1 : k + 1] = np.copy(self.basis[i : k])
                        self.basis[i] = np.copy(v)
                        self.B, self.mu = self.GSO(mode = "square")
                    k = max(i - 1, 0)
            k += 1
        
        return self


class random_lattice(lattice):
    def __init__(self, n: int):
        while True:
            lattice.basis = np.random.randint(1000, size = (n, n))
            if np.linalg.det(lattice.basis) != 0: break
        lattice.nrows = n
        lattice.ncols = n
        lattice.mu = np.eye(n)
        lattice.basis_star = np.zeros((n, n))
        lattice.B = np.zeros(n)
    
