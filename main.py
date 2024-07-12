import latalg
import numpy as np

def main():
    b = latalg.random_lattice(4)
    print(b.DeepLLL().basis)

if __name__ == '__main__':
    main()

