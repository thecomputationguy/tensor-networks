import numpy as np
from mps import mps_operations as mps

# main function

def main():
    # define wave vector
    L = 20
    chi_max = 10

    psi = np.random.rand(2**L)
    psi = psi / np.linalg.norm(psi)
    #print(psi)
    
    # compress the wave vector to a MPS
    Ms = mps.compress(psi, L, chi_max)
    #print(Ms)
    
    # Check if MPS is close to the original state
    psi_mps = mps.reconstruct_psi_from_mps(Ms)
    #print(psi_mps)

    # Find relative error of the reconstruction
    fidelity = np.abs(np.vdot(psi, psi_mps))**2
    print("Fidelity: ", fidelity)

    # Compare size of the MPS with the original state
    size_psi = psi.size
    size_mps = sum([M.size for M in Ms])

    print("Size of the original state: ", size_psi)
    print("Size of the MPS: ", size_mps)
    print("Compression factor: ", size_psi / size_mps)


if __name__ == "__main__":
    main()