import PLL
import numpy as np

def main():
    b = PLL.random_lattice(3)
    print(np.linalg.norm(b.basis[0]))
    print(np.linalg.norm(b.DeepLLL()[0]))

if __name__ == '__main__':
    main()

