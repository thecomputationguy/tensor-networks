import numpy as np

def split(M, bond_dim):
    """
    Split a matrix M via SVD and keep only the first bond_dim singular values.
    """

    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    Vd = Vd.reshape(bonds, 2, -1)
    U = U.reshape((-1, 2, bonds))

    # keep only chi bonds
    chi = np.min([bond_dim, bonds])
    U, S, Vd = U[:, :, :chi], S[:chi], Vd[:chi]

    return U, S, Vd

def dense_to_mps(psi, bond_dim, num_sites):
    """
    Turn a dense state vector psi into an MPS with bond dimension bond_dim.
    """
    Ms = []
    Ss = []

    psi = np.reshape(psi, (2, -1))
    U, S, Vd = split(psi, bond_dim)

    Ms.append(U)
    Ss.append(S)
    bondL = Vd.shape[0]
    psi = np.tensordot(np.diag(S), Vd, 1)

    for _ in range(num_sites - 2):
        psi = np.reshape(psi, (2*bondL, -1))
        U, S, Vd = split(psi, bond_dim)
        Ms.append(U)
        Ss.append(S)

        psi = np.tensordot(np.diag(S), Vd, 1)
        bondL = Vd.shape[0]

    # Last tensor
    psi = np.reshape(psi, (-1, 1))
    U, _, _ = np.linalg.svd(psi, full_matrices=False)

    U = np.reshape(U, (-1, 2, 1))
    Ms.append(U)

    return Ms, Ss