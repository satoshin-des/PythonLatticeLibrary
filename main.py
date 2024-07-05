import latalg
import numpy as np

def main():
    b = latalg.lattice(np.array([[20, 5], [90, 0]]))

    print(f"{b.DeepLLL()}")

if __name__ == '__main__':
    main()
