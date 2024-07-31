"""
# PLM(Python Lattice Module)

PLM(official name: Python Lattice Module) is a module for lattices.

Provides
- lattices and operations on lattices
- lattice reduction algorithms
- other lattice algorithms(e.g. SVP, CVP, and more)
"""

import numpy as np
import sys

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


    def print(self):
        """Prints lattice basis.
        """
        print(self.basis)


    def vol(self) -> float:
        """Computes volume of lattice

        Returns:
            float: lattice
        """
        return np.sqrt(np.linalg.det(self.basis * self.basis.T))
    

    def volume(self) -> float:
        """Computes volume of lattice

        Returns:
            float: lattice
        """
        return np.sqrt(np.linalg.det(self.basis * self.basis.T))


    def dual(self):
        """Computes dual-lattice.

        Returns:
            lattice: Dual-lattice.
        """
        c = lattice(self.basis)
        self.basis_star, self.mu = self.GSO()
        c.basis = np.matmul(np.matmul(np.linalg.inv(self.mu.T), np.linalg.inv(np.matmul(self.basis_star, self.basis_star.T))), self.basis_star)
        c.basis_star, c.mu = c.GSO()
        return c


    def GSO(self, mode: str = "normal") -> np.ndarray:
        """Gram-Schmidt's method.

        Args:
            mode (str, optional): What value this function returns. Defaults to "normal".

        - mode = "normal"

            Returns:
                np.ndarray: GSO-vectors, GSO coefficient matrix
        - mode = "square"

            Returns:
                np.ndarray: Squared norms of GSO-vectors, GSO coefficient matrix
        - mode = "both"

            Returns:
                np.ndarray: GSO-vectors, Squared norms of GSO-vectors, GSO coefficient matrix
        """
        for i in range(self.nrows):
            self.basis_star[i] = np.copy(self.basis[i])
            for j in range(i):
                self.mu[i, j] = np.dot(self.basis[i], self.basis_star[j]) / np.dot(self.basis_star[j], self.basis_star[j])
                self.basis_star[i] -= self.mu[i, j] * np.copy(self.basis_star[j])
            if mode == "square" or mode == "both":
                self.B[i] = np.dot(self.basis_star[i], self.basis_star[i])
        if mode == "normal":
            return self.basis_star, self.mu
        elif mode == "square":
            return self.B, self.mu
        elif mode == "both":
            return self.basis_star, self.B, self.mu
    

    def potential(self) -> float:
        """Computes potential of the lattice basis

        Returns:
            float: Potential
        """
        return np.prod(self.B ** np.arange(self.nrows, 0. -1))


    def pot(self) -> float:
        """Computes potential of the lattice basis.

        Returns:
            float: Potential.
        """
        return np.prod(self.B ** np.arange(self.nrows, 0. -1))


    def SS(self) -> float:
        """Computes the squared-sum of the lengths of the GSO-vectors.

        Returns:
            float: The squared-sum of the lengths of the GSO-vectors.
        """
        return np.sum(self.GSO(mode = "square")[0])
    

    def SquaredSum(self) -> float:
        """Computes the squared-sum of the lengths of the GSO-vectors.

        Returns:
            float: The squared-sum of the lengths of the GSO-vectors.
        """
        return np.sum(self.GSO(mode = "square")[0])


    def orthogonality_defect(self) -> float:
        """Computes orthogonality defect of the lattice basis.

        Returns:
            float: Orthogonality defect.
        """
        return np.prod(np.sum(self.basis * self.basis, axis = 1)) / self.vol()


    def size(self) -> np.ndarray:
        """Size-reduction(algorithm is from M. Yasuda and Y. Aoki).

        Returns:
            np.ndarray: Size-reduced lattice basis matrix.
        """
        for i in range(1, self.nrows):
            for j in range(i)[::-1]:
                if self.mu[i, j] > 0.5 or self.mu[i, j] < -0.5:
                    q = round(self.mu[i, j])
                    self.basis[i] -= q * np.copy(self.basis[j])
                    self.mu[i, : j + 1] -= q * np.copy(self.mu[j, : j + 1])
        return self.basis


    def Gauss(self) -> np.ndarray:
        """Gauss reduction

        Returns:
            np.ndarray: Gauss reduced lattice basis matrix.
        """
        if self.nrows > 2:
            sys.exit("Rank Error: Lagrange reduction is only for 2-dimensional lattices")
        if np.linalg.norm(self.basis[0]) > np.linalg.norm(self.basis[1]):
            self.basis[0], self.basis[1] = np.copy(self.basis[1]), np.copy(self.basis[0])
        while True:
            v = self.basis[1] - round(np.dot(self.basis[0], self.basis[1]) / np.dot(self.basis[0], self.basis[0])) * self.basis[0]
            self.basis[1] = self.basis[0]
            self.basis[0] = np.copy(v)
            if np.linalg.norm(self.basis[0]) >= np.linalg.norm(self.basis[1]):
                self.basis[0], self.basis[1] = np.copy(self.basis[1]), np.copy(self.basis[0])
                return self.basis


    def Lagrange(self) -> np.ndarray:
        """Lagrange reduction. This is other name of Gauss().

        Returns:
            np.ndarray: Lagrange reduced lattice basis matrix.
        """
        if self.nrows > 2:
            sys.exit("Rank Error: Lagrange reduction is only for 2-dimensional lattices")
        if np.linalg.norm(self.basis[0]) > np.linalg.norm(self.basis[1]):
            self.basis[0], self.basis[1] = np.copy(self.basis[1]), np.copy(self.basis[0])
        while True:
            v = self.basis[1] - round(np.dot(self.basis[0], self.basis[1]) / np.dot(self.basis[0], self.basis[0])) * self.basis[0]
            self.basis[1] = self.basis[0]
            self.basis[0] = np.copy(v)
            if np.linalg.norm(self.basis[0]) >= np.linalg.norm(self.basis[1]):
                self.basis[0], self.basis[1] = np.copy(self.basis[1]), np.copy(self.basis[0])
                return self.basis


    def LLL(self, delta: float = 0.99) -> np.ndarray:
        """LLL-reduction (algorithm is from A. K. Lenstra, H. W. Lenstra, and L. Lovasz (1982)).

        Args:
            delta (float, optional): Reduction parameter(0.5 <= delta <= 1). Defaults to 0.99.

        Returns:
            np.ndarray: LLL-reduced basis
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
        return self.basis
    

    def DeepLLL(self, delta : float = 0.99) -> np.ndarray:
        """Deep-LLL-reduction (algorithm is from C.P.Schnorr and M. Euchner(1994)).

        Args:
            delta (float, optional): Reduction parameter. Defaults to 0.99.

        Returns:
            np.ndarray: Deep-LLL-reduced basis
        """
        self.B, self.mu = self.GSO(mode = "square")
        k = 1
        while k < self.nrows:
            for j in range(k)[::-1]:
                if abs(self.mu[k, j]) > 0.5:
                    q = round(self.mu[k, j])
                    self.basis[k] -= q * np.copy(self.basis[j])
                    self.mu[k, : j + 1] -= q * np.copy(self.mu[j, : j + 1])
            C = np.dot(self.basis[k], self.basis[k])
            i = 0
            while i < k:
                if C >= delta * self.B[i]:
                    C -= self.mu[k, i] * self.mu[k, i] * self.B[i]
                    i += 1
                else:
                    t = np.copy(self.basis[k])
                    self.basis[i + 1 : k + 1] = np.copy(self.basis[i : k])
                    self.basis[i] = np.copy(t)
                    self.B, self.mu = self.GSO(mode = "square")
                    k = max(i - 1, 0)
            k += 1
        return self.basis
    

    def DEEP(self, delta: float = 0.99) -> np.ndarray:
        """Other name of DeepLLL().

        Args:
            delta (float, optional): Reduction parameter. Defaults to 0.99.

        Returns:
            np.ndarray: Deep-LLL-reduced lattice basis matrix.
        """
        return self.DeepLLL(delta = delta)


    def PotLLL(self, delta: float = 0.99) -> np.ndarray:
        """Potential-LLL-reduces the lattice basis matrix.

        Args:
            delta (float, optional): Reduction parameter. Defaults to 0.99.

        Returns:
            np.ndarray: A potential-LLL-reduced lattice basis matrix.
        """
        l = 0
        self.basis = self.LLL(delta = 0.99)
        self.B, self.mu = self.GSO(mode = "square")
        while l < self.nrows:
            for j in range(l)[::-1]:
                if abs(self.mu[l, j]) > 0.5:
                    q = round(self.mu[l, j])
                    self.basis[l] -= q * np.copy(self.basis[j])
                    self.mu[l, : j + 1] -= q * np.copy(self.mu[j, : j + 1])
            P = P_min = 1.; k = 0
            for j in range(l)[::-1]:
                P *= (self.B[l] + np.sum(self.mu[l, j: l] * self.mu[l, j: l] * self.B[j: l])) / self.B[j]
                if P < P_min: k = j; P_min = P
            if delta > P_min:
                t = np.copy(self.basis[l])
                self.basis[k + 1: l + 1] = np.copy(self.basis[k: l])
                self.basis[k] = np.copy(t)
                self.B, self.mu = self.GSO(mode = "square")
                l = k
            else: l += 1
        return self.basis


    def DualDeepLLL(self, delta : float = 0.99) -> np.ndarray:
        """Dual Deep-LLL-reduction.

        Args:
            delta (float, optional): Reduction parameter. Defaults to 0.99.

        Returns:
            np.ndarray: Dual Deep-LLL-reduced basis.
        """
        self.B, self.mu = self.GSO(mode = "square")
        nu = np.zeros((self.nrows, self.nrows))
        nu[self.nrows - 1, self.nrows - 1] = 1
        k = self.nrows - 2
        while k >= 0:
            nu[k, k] = 1
            for j in range(k + 1, self.nrows):
                nu[k, j] = -np.sum(self.mu[j, k: j] * nu[k, k: j])
            if nu[k, j] > 0.5 or nu[k, j] < -0.5:
                q = round(nu[k, j])
                self.basis[j] += q * np.copy(self.basis[k])
                nu[k, j: self.nrows] -= q * np.copy(nu[j, j: self.nrows])
                self.mu[j, : k + 1] += q * np.copy(self.mu[k, : k + 1])
            l = self.nrows - 1
            D = np.sum(nu[k, k:] * nu[k, k:] / self.B[k:])
            while l > k:
                if self.B[l] * D < delta:
                    t = np.copy(self.basis[k])
                    self.basis[k: l] = np.copy(self.basis[k + 1: l + 1])
                    self.basis[l] = np.copy(t)
                    k = min(l, self.nrows - 2) + 1
                    self.B, self.mu = self.GSO(mode = "square")
                else:
                    D -= nu[k, l] * nu[k, l] / self.B[l]
                    l -= 1
            k -= 1
        return self.basis
    

    def ENUM_SVP(self) -> np.ndarray:
        """Enumerates the shortest vector on the lattices(algorithm is from N. Gama, P. Q. Nguyen and O. Regev(2010)).

        Returns:
            np.ndarray: The shortest vector
        """
        def _ENUM_(mu: np.ndarray, B: np.ndarray, n: int, delta: float) -> np.ndarray:
            """A sub-routine of the function ENUM. This computes a vector whose norm is short(algorithm is from N. Gama, P. Q. Nguyen and O. Regev(2010)).
    
            Args:
                mu (np.ndarray): GSO-coefficient matirx
                B (np.ndarray): Squared norms of GSO-vectors.
                n (int): Rank of lattice.
                delta (float): A parameter.

            Returns:
                np.ndarray: A vector whose norm is shorter than delta * B[0]
            """
            sigma = np.zeros((n + 1, n), float); rho = np.zeros(n + 1, float)
            r = np.arange(n + 1, dtype=int); r -= 1; r = np.roll(r, -1)
            v = np.zeros(n, int); v[0] = 1
            c = np.zeros(n); w = np.zeros(n, int)
            last_nonzero = 0; k = 0; R = delta * B[0]
            while True:
                if R < 1: return np.zeros(n, int)
                tmp = v[k] - c[k]; tmp *= tmp
                rho[k] = rho[k + 1] + tmp * B[k]
                if rho[k] <= R:
                    if k == 0: return v
                    k -= 1
                    r[k - 1] = max(r[k - 1], r[k])
                    for i in range(k + 1, r[k] + 1)[::-1]:
                        sigma[i, k] = sigma[i + 1, k] + mu[i, k] * v[i]
                    c[k] = -sigma[k + 1, k]
                    v[k] = np.round(c[k])
                    w[k] = 1
                else:
                    k += 1
                    if k == n: return None
                    r[k - 1] = k
                    if k >= last_nonzero:
                        last_nonzero = k
                        v[k] += 1
                    else:
                        if v[k] > c[k]:	v[k] -= w[k]
                        else: v[k] += w[k]
                        w[k] += 1
        
        self.B, self.mu = self.GSO(mode = "square")
        ENUM_v = np.zeros(self.nrows, int)
        delta = 1.
        while True:
            pre_ENUM_v = np.copy(ENUM_v)
            ENUM_v = _ENUM_(self.mu, self.B, self.nrows, delta)
            if ENUM_v is None: return np.matmul(pre_ENUM_v, self.basis)
            delta *= 0.99
    

    def SVP(self) -> np.ndarray:
        """Enumerates the shortest vector on the lattices.

        Returns:
            np.ndarray: The shortest vector on the lattice.
        """
        return self.ENUM_SVP()


    def project_basis(self, k: int, l: int) -> np.ndarray:
        """Computes a projected lattice basis matrix.

        Args:
            k (int): _description_
            l (int): _description_

        Returns:
            np.ndarray: Projected basis.
        """
        self.basis_star, self.mu = self.GSO()
        pi_b = np.zeros((l - k + 1, self.ncols))
        for i in range(k, l + 1):
            for j in range(k, self.nrows):
                pi_b[i - k] += np.dot(self.basis[i], self.basis_star[j]) / np.dot(self.basis_star[j], self.basis_star[j]) * np.copy(self.basis_star[j])
        return pi_b


    def Babai(self, w: np.ndarray):
        """Computes an approximate solution of CVP for target w using Babai's nearest plane algorithm(algorithm is from L. Babai(1986)).

        Args:
            w (np.ndarray): A target vector.
        """
        t = np.copy(w)
        self.basis_star, self.mu = self.GSO(mode = "normal")
        for i in range(self.nrows)[::-1]:
            c = round(np.dot(t, self.basis_star[i]) / np.dot(self.basis_star[i], self.basis_star[i]))
            t -= c * np.copy(self.basis[i])
        return w - t
    

    def ENUM_CVP(self, t: np.ndarray) -> np.ndarray:
        """Enumerates the closest vector to target t on the lattice(algorithm is from M. Liu and P. Q. Nguyen(2013)).

        Args:
            t (np.ndarray): A target vector.

        Returns:
            np.ndarray: The closest vector.
        """
        def _ENUM_(mu: np.ndarray, B: np.ndarray, n: int, a: np.ndarray, R: float) -> np.ndarray:
            """A sub-routine of function ENUM. This computes a vector that is close to target(algorithm is from M. Liu and P. Q. Nguyen(2013)).

            Args:
                mu (np.ndarray): GSO-coefficient matrix.
                B (np.ndarray): Squared norms of GSO-vectors.
                n (int): Rank of lattice.
                a (np.ndarray): The coefficient vector of target vector on the lattice basis.
                R (float): Upper bound of enumeration.

            Returns:
                np.ndarray: A vector that is close to target.
            """
            sigma = np.zeros((n + 1, n), float); rho = np.zeros(n + 1, float)
            r = np.arange(n + 1, dtype=int); r -= 1; r = np.roll(r, -1)
            v = np.zeros(n, int)
            c = np.zeros(n); w = np.zeros(n, int)
            for k in range(n)[::-1]:
                for i in range(k + 1, n)[::-1]:
                    sigma[i, k] = sigma[i + 1, k] + (a[i] - v[i]) * mu[i, k]
                c[k] = a[k] + sigma[k + 1, k]
                v[k] = round(c[k]); w[k] = 1
                tmp = c[k] - v[k]; tmp *= tmp
                rho[k] = rho[k + 1] + tmp * B[k]
            k = 0
            while True:
                if R < 1: return np.zeros(n, int)
                tmp = v[k] - c[k]; tmp *= tmp
                rho[k] = rho[k + 1] + tmp * B[k]
                if rho[k] <= R:
                    if k == 0: return v
                    k -= 1
                    r[k - 1] = max(r[k - 1], r[k])
                    for i in range(k + 1, r[k] + 1)[::-1]:
                        sigma[i, k] = sigma[i + 1, k] + (a[i] - v[i]) * mu[i, k]
                    c[k] = a[k] + sigma[k + 1, k]
                    v[k] = np.round(c[k])
                    w[k] = 1
                else:
                    k += 1
                    if k == n: return None
                    r[k - 1] = k
                    if v[k] > c[k]:	v[k] -= w[k]
                    else: v[k] += w[k]
                    w[k] += 1
        
        self.B, self.mu = self.GSO(mode = "square")
        ENUM_v = np.zeros(self.nrows, int)
        babaivec = self.Babai(t); R = np.dot(babaivec - t, babaivec - t)
        a = np.matmul(t, np.linalg.pinv(self.basis))
        while True:
            pre_ENUM_v = np.copy(ENUM_v)
            ENUM_v = _ENUM_(self.mu, self.B, self.nrows, a, R)
            if ENUM_v is None: return np.matmul(pre_ENUM_v, self.basis)
            R *= 0.99
    

    def CVP(self, t: np.array) -> np.ndarray:
        return self.ENUM_CVP(t)


class random_lattice(lattice):
    def __init__(self, n: int):
        lattice.basis = np.eye(n, dtype = int)
        lattice.basis[:, 0] = np.random.randint(500, 1000, size = (n, ))
        lattice.nrows = n
        lattice.ncols = n
        lattice.mu = np.eye(n)
        lattice.basis_star = np.zeros((n, n))
        lattice.B = np.zeros(n)
    
