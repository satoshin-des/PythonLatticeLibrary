import PLL
import numpy as np
import sympy

def main():
    b = PLL.random_lattice(60)
    print(np.linalg.norm(b.basis[0]))
    t = b.LLL()[0]; print(t, np.linalg.norm(t))
    #t = PLL.lattice.ENUM(b); print(t @ b.basis, np.linalg.norm(t @ b.basis))
    t = b.BKZ(15)[0]; print(t, np.linalg.norm(t))


if __name__ == '__main__':
    main()

