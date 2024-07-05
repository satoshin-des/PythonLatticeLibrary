import latalg
import numpy as np

def main():
    b = latalg.random_lattice(7)
    print(f"{b.LLL()}")

if __name__ == '__main__':
    main()

