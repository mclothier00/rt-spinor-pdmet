import numpy as np
from pyscf import scf, gto, ao2mo, fci
from scipy import linalg
import itertools


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print("{:.5f}".format(matrix[i][j]), end=" ")
        print(f"\n")


def reshape_rtog_matrix(a):
    ## reshape a block diagonal matrix a to a generalized form with 1a,1b,2a,2b, etc.

    num_rows, num_cols = a.shape
    block_indices = np.arange(num_cols)
    spin_block_size = int(num_cols / 2)

    alpha_block = block_indices[:spin_block_size]
    beta_block = block_indices[spin_block_size:]

    indices = [list(itertools.chain(i)) for i in zip(alpha_block, beta_block)]

    indices = np.asarray(indices).reshape(-1)

    new_a = a[:, indices]

    return new_a


def reshape_rtog_tensor(a):
    ## reshape a block diagonal tensor a to a generalized form with columns as 1a,1b,2a,2b, etc.

    num_rows, num_cols, dim1, dim2 = a.shape
    block_indices = np.arange(num_cols)
    spin_block_size = int(num_cols / 2)

    alpha_block = block_indices[:spin_block_size]
    beta_block = block_indices[spin_block_size:]

    indices = [list(itertools.chain(i)) for i in zip(alpha_block, beta_block)]

    indices = np.asarray(indices).reshape(-1)

    new_a = a[:, :, :, indices]

    return new_a


### initialize system ###

U = 3.0
j1 = 0.4
j2 = 0.5
j3 = 0.6
j4 = 0.75
t1 = 0.7
t2 = 0.8
t3 = 0.2
t4 = 0.3
norb = 6
nelec = 3

# block diagonal
h1e = [
    [-1.5, -t1, 0, 0, -j1, 0],
    [-t1, -1.0, -t2, -j2, 0, -j3],
    [0, -t2, 1.0, 0, -j4, 0],
    [0, -j2, 0, -0.65, -t3, 0],
    [-j1, 0, -j4, -t3, -0.25, -t4],
    [0, -j3, 0, 0, -t4, 0.65],
]


#### diagonalize hamiltonians ###
#
h1_eig, eigvec = linalg.eig(h1e)
#
print(f"h1 eig: {h1_eig}")

### GHF check ###
spatial = int(norb / 2)
h1e = np.asarray(h1e)
eri = np.zeros((spatial, spatial, spatial, spatial))
ovlp = np.eye(norb)

mol = gto.M()
mol.nelectron = nelec
mol.incore_anyway = True
mol.unit = "au"

print(f"initial h1e: \n {h1e}")

mf = scf.GHF(mol)
mf.init_guess = "hcore"
mf.get_hcore = lambda *args: h1e
mf.get_ovlp = lambda *args: ovlp
mf._eri = ao2mo.restore(1, eri, spatial)
mf.kernel()

ener_tot = mf.energy_tot()
print(f"total E from GHF: {ener_tot}")
print(f"orbital energies: {mf.mo_energy}")


### fci test ###

eri = np.zeros((norb, norb, norb, norb))
eri = ao2mo.restore(1, eri, norb)
# print(f'shape of eri: {eri.shape}')
h1e = mf.get_hcore()
print(f"h1e from mf.get_hcore \n {h1e}")


### testing reshaping matrices and tensors
eri = reshape_rtog_tensor(eri)
h1e = reshape_rtog_matrix(h1e)
print(f"reshaped h1e: \n {h1e}")


# norb =2
e, fcivec = fci.fci_dhf_slow.kernel(h1e, eri, norb, nelec, nroots=20)
# print(f'ci coeff: {fcivec}')
print(f"fci energy: {e}")

# h_diag = fci.fci_dhf_slow.make_hdiag(h1e, eri, norb, nelec)
# print(f'h diag: {h_diag}')
#
# tran = np.transpose(np.vstack(fcivec))
# fci_ham_diag = np.diag(e)
#
# fci_ham = np.dot(tran, np.dot(fci_ham_diag, np.transpose(tran)))
# print('created hamiltonian:')
# print_matrix(fci_ham)
#
# e_new, vec_new = linalg.eigh(fci_ham)
# print(f'fci energies from eigh: \n {e_new}')
# print('fci vectors from eigh:')
# print_matrix(vec_new)
#
