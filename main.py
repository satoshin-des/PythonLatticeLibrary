import PLM
import numpy as np
import sympy

def main():
    b = PLM.random_lattice(40)
    print(b.basis)
    a = b.PotLLL()[0]; print(a, np.linalg.norm(a))
    #a = PLM.lattice.ENUM(b); print(a @ b.basis, np.linalg.norm(a @ b.basis))
    #a = b.BKZ(15); print(a, np.linalg.norm(a[0]))
    t = np.random.randint(10, 20, size=40)
    print(t)
    print(b.ENUM_CVP(t))


if __name__ == '__main__':
    main()

