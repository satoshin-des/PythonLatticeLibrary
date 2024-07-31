import PLM
import numpy as np
import sympy

def main():
    b = PLM.random_lattice(20)
    print(b.basis)
    #t = b.PotLLL()[0]; print(t, np.linalg.norm(t))
    #t = PLM.lattice.ENUM(b); print(t @ b.basis, np.linalg.norm(t @ b.basis))
    t = b.BKZ(15); print(t, np.linalg.norm(t[0]))


if __name__ == '__main__':
    main()

