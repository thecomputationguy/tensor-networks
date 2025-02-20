import numpy as np
import scipy

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

def compress(psi, L, chi_max):
    """
    Compress an wave vector psi to a MPS with bond dimension chi_max.
    """
    psi_aR = np.reshape(psi, (1, 2**L)) # reshape initial state to matrix with a trivial left bond dimension of 1
    Ms = []
    for n in range(1, L+1):
        chi_n, dim_R = psi_aR.shape
        assert dim_R == 2**(L-n), f"Dimension mismatch: dim_R = {dim_R}, 2**(L-n) = {2**(L-n)}"
        psi_LR = np.reshape(psi_aR, (chi_n * 2, dim_R // 2)) # reshaping psi_aR into a tensor of dimensions ...
        # ... (current left bond chi, 2 for qubits, remaining unprocessed right state)
        M_n, lambda_n, psi_tilde = scipy.linalg.svd(psi_LR, full_matrices=False, lapack_driver='gesvd') # SVD of reshaped state

        # Truncate the bond dimension by keeping only the chi_max largest singular values
        if len(lambda_n) > chi_max:
            keep = np.argsort(lambda_n)[::-1][:chi_max]
            lambda_n = lambda_n[keep]
            psi_tilde = psi_tilde[keep, :]

        # Reshape the singular values into a matrix to get the MPS tensor for site n
        chi_np1 = len(lambda_n)
        M_n = np.reshape(M_n, (chi_n, 2, chi_np1)) # M_n should be of dimensons (chi_n, 2, chi_np1)
        Ms.append(M_n)

        # Prepare the rest of the state to the right
        psi_aR = np.tensordot(np.diag(lambda_n), psi_tilde, (1, 0))

    # Last tensor
    assert psi_aR.shape == (chi_np1, 2)

    return Ms

def reconstruct_psi_from_mps(Ms):
    """
    Reconstruct the wave vector psi from a MPS.
    """
    psi = Ms[0].reshape(-1, 2)
    for M in Ms[1:]:
        psi = np.tensordot(psi, M, axes=([1], [0]))

    return psi.ravel()
