import PLL
import numpy as np

def main():
    b = PLL.random_lattice(4)
    print(b.DualDeepLLL().basis)

if __name__ == '__main__':
    main()

