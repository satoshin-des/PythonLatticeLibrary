import main
import numpy as np

def pmain():
    b = main.lattice(np.array([[20, 5], [90, 0]]), 2, 2)
    
    print(f"gso = {b.GSO()}")


pmain()

