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
j1 = 0.8 + 0.2j
j2 = 0.4
t1 = 0.5
t2 = 0.2
norb = 4
nelec = 2 # number of electrons

h1e = [
        [-0.6, -t1, 0, -j1],
        [-t1, 0.6, -j2, 0],
        [0, -j2, -0.4, -t2],
        [-j1, 0, -t2, 0.4]  
        ]

fci_ham = [
            [0, -j2, 0, 0, j1, 0],
            [-j2, -1.0, -t1, -t2, 0, j1],
            [0, -t1, 0.2, 0, -t2, 0],
            [0, -t2, 0, -0.2, -t1, 0],
            [j1, 0, -t2, -t1, 1.0, -j2],
            [0, j1, 0, 0, -j2, 0]
            ]

fci_ham1 = [
            [0, -0.4, 0, 0, -0.8, 0],
            [-0.4, -1.0, 0.5, -0.2, 0, -0.8],
            [0, 0.5, 0.2, 0, -0.2, 0],
            [0, -0.2, 0, -0.2, 0.5, 0],
            [-0.8, 0, -0.2, 0.5, 1.0, -0.4],
            [0, -0.8, 0, 0, -0.4, 0]]

fci_ham2 = [
            [-1.0, t1, -t2, -j2, -j1, 0],
            [t1, -0.6, 0, 0, 0, -t2],
            [-t2, 0, -0.3, 0, 0, t1],
            [-j2, 0, 0, 0.3, 0, -j1],
            [-j1, 0, 0, 0, 0.6, -j2],
            [0, -t2, t1, -j1, -j2, 1.0]
            ]

print(f'my hamiltonian:\n {np.asarray(fci_ham)}')

### diagonalize hamiltonians ###

h1_eig, eigvec = linalg.eig(h1e)

print(f'h1 eig: {h1_eig}')

fci_eig, fci_eigvec = linalg.eigh(fci_ham)

print(f'fci eig: {fci_eig}')
print(f'fci eigenvectors: {np.transpose(fci_eigvec)}')


### GHF check ###

h1e = np.asarray(h1e)
eri = np.zeros((2, 2, 2, 2))
#for i in range(0,2):
#    eri[i,i,i,i] = U
ovlp = np.eye(norb)

mol = gto.M()
mol.nelectron = nelec
mol.incore_anyway = True
mol.unit = 'au'

mf = scf.GHF(mol)
mf.init_guess = 'hcore'
mf.get_hcore = lambda *args: h1e
mf.get_ovlp = lambda *args: ovlp
mf._eri = ao2mo.restore(1, eri, 2)
mf.kernel()

ener_tot = mf.energy_tot()
print(f'total E from GHF: {ener_tot}')
print(f'orbital energies: {mf.mo_energy}')

### fci test ###

eri = np.zeros((norb,norb,norb,norb), dtype=complex)
for i in range(0,norb):
    eri[i,i,i,i] = U
#eri = ao2mo.restore(1, eri, norb)
#print(f'eri element: {eri[1,1,1,1]}')

h1e = mf.get_hcore()
print(h1e)

e, fcivec = fci.fci_dhf_slow.kernel(h1e, eri, norb, nelec, nroots=6)
print(f'fci values: {fcivec}')
print(f'fci energy: {e}')

h_diag = fci.fci_dhf_slow.make_hdiag(h1e, eri, norb, nelec)
print(f'h diag: {h_diag}')

tran = np.transpose(np.vstack(fcivec))
fci_ham_diag = np.diag(e)

fci_ham = np.dot(tran, np.dot(fci_ham_diag, np.transpose(tran)))
print(f'created hamiltonian \n')
print_matrix(fci_ham)

e_new, vec_new = linalg.eigh(fci_ham)
print(e_new)
print(vec_new)

