import PLM
import numpy as np
import sympy

def main():
    b = PLM.random_lattice(70)
    print(b.basis)
    #a = b.PotLLL()[0]; print(a, np.linalg.norm(a))
    #a = PLM.lattice.ENUM(b); print(a @ b.basis, np.linalg.norm(a @ b.basis))
    t = np.random.randint(10, 20, size=70)
    print(t)
    print(b.ENUM_CVP(t))
    print(b.ENUM_SVP())


if __name__ == '__main__':
    main()

