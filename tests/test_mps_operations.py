import numpy as np
import scipy
import pytest
from mps.mps_operations import split, dense_to_mps, compress, reconstruct_psi_from_mps

def test_split():
    M = np.random.rand(4, 4)
    U, S, Vd = split(M, 2)
    assert U.shape == (4, 2, 2)
    assert S.shape == (2,)
    assert Vd.shape == (2, 2, 2)

def test_dense_to_mps():
    psi = np.random.rand(4)
    psi = psi / np.linalg.norm(psi)
    Ms, Ss = dense_to_mps(psi, 2, 3)

    assert len(Ms) == 3
    assert len(Ss) == 2
    assert Ms[0].shape == (2, 2, 2)
    assert Ms[1].shape == (4, 2, 2)
    assert Ms[2].shape == (2, 2, 2)

    # Check that the matrices are orthonormal
    assert np.allclose(np.einsum('ijk, ijk', Ms[0], Ms[0]), np.eye(2))
    assert np.allclose(np.einsum('ijk, ijk', Ms[1], Ms[1]), np.eye(2))
    assert np.allclose(np.einsum('ijk, ijk', Ms[2], Ms[2]), np.eye(2))
    
def test_compress():
    psi = np.random.rand(8)
    psi = psi / np.linalg.norm(psi)
    Ms, Ss = compress(psi, 3, 2)

    # Check that the bond dimensions are correct
    assert len(Ms) == 3
    assert len(Ss) == 2

    # Check that the shapes are correct
    assert Ms[0].shape == (1, 2, 2)
    assert Ms[1].shape == (2, 2, 2)
    assert Ms[2].shape == (2, 2, 1)

    assert Ss[0].shape == (2,)
    assert Ss[1].shape == (2,)
    assert Ss[2].shape == (1,)

    # Check that the matrices are orthonormal
    assert np.allclose(np.einsum('ijk, ijk', Ms[0], Ms[0]), np.eye(2))
    assert np.allclose(np.einsum('ijk, ijk', Ms[1], Ms[1]), np.eye(2))
    assert np.allclose(np.einsum('ijk, ijk', Ms[2], Ms[2]), np.eye(2))

def test_reconstruct_psi_from_mps():
    psi = np.random.rand(8)
    psi = psi / np.linalg.norm(psi)

    Ms = compress(psi, 3, 2)
    psi_reconstructed = reconstruct_psi_from_mps(Ms)

    assert np.allclose(psi, psi_reconstructed)