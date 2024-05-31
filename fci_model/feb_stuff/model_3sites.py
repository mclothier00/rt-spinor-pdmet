import numpy as np
from pyscf import scf, gto, ao2mo, fci
from scipy import linalg

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print("{:.5f}".format(matrix[i][j]), end=' ')
        print(f'\n')

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

h1e = [
        [-1.5, -t1, 0, 0, -j1, 0],
        [-t1, -1.0, -t2, -j2, 0, -j3],
        [0, -t2, 1.0, 0, -j4, 0],
        [0, -j2, 0, -0.65, -t3, 0],
        [-j1, 0, -j4, -t3, -0.25, -t4],
        [0, -j3, 0, 0, -t4, 0.65],
        ]

#fci_ham = [
#            [0, -j2, 0, 0, j1, 0],
#            [-j2, -1.0, -t1, -t2, 0, j1],
#            [0, -t1, 0.2, 0, -t2, 0],
#            [0, -t2, 0, -0.2, -t1, 0],
#            [j1, 0, -t2, -t1, 1.0, -j2],
#            [0, j1, 0, 0, -j2, 0]
#            ]
#
#print(f'my hamiltonian:\n {np.asarray(fci_ham)}')
#
#### diagonalize hamiltonians ###
#
h1_eig, eigvec = linalg.eig(h1e)
#
print(f'h1 eig: {h1_eig}')
#
#fci_eig, fci_eigvec = linalg.eigh(fci_ham)
#
#print(f'fci eig: {fci_eig}')
#print(f'fci eigenvectors: {np.transpose(fci_eigvec)}')


### GHF check ###
spatial = int(norb/2)
h1e = np.asarray(h1e)
eri = np.zeros((spatial, spatial, spatial, spatial))
ovlp = np.eye(norb)

mol = gto.M()
mol.nelectron = nelec
mol.incore_anyway = True
mol.unit = 'au'

mf = scf.GHF(mol)
mf.init_guess = 'hcore'
mf.get_hcore = lambda *args: h1e
mf.get_ovlp = lambda *args: ovlp
mf._eri = ao2mo.restore(1, eri, spatial)
mf.kernel()

ener_tot = mf.energy_tot()
print(f'total E from GHF: {ener_tot}')
print(f'orbital energies: {mf.mo_energy}')


### fci test ###

eri = np.zeros((norb,norb,norb,norb))
eri = ao2mo.restore(1, eri, norb)
print(f'shape of eri: {eri.shape}')
h1e = mf.get_hcore()
print(h1e)
#norb =2
e, fcivec = fci.fci_dhf_slow.kernel(h1e, eri, norb, nelec, nroots=20)
print(f'fci values: {fcivec}')
print(f'fci energy: {e}')

h_diag = fci.fci_dhf_slow.make_hdiag(h1e, eri, norb, nelec)
print(f'h diag: {h_diag}')

tran = np.transpose(np.vstack(fcivec))
fci_ham_diag = np.diag(e)

fci_ham = np.dot(tran, np.dot(fci_ham_diag, np.transpose(tran)))
print('created hamiltonian:')
print_matrix(fci_ham)

e_new, vec_new = linalg.eigh(fci_ham)
print(f'fci energies from eigh: \n {e_new}')
print('fci vectors from eigh:')
print_matrix(vec_new)

