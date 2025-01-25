import numpy as np
from pyscf import scf, gto, fci, ao2mo
from functools import reduce

mol = gto.M(
    verbose = 0,
    output = None,
    #atom='Li 0 0 0',
    atom =[
    ['H', ( 1.,-1.    , 0.   )],
    ['H', ( 0.,-1.    ,-1.   )],
    ['H', ( 1.,-0.5   ,-1.   )],
    ['H', ( 0.,-0.    ,-1.   )]
    #['H', ( 1.,-0.5   , 0.   )],
    #['H', ( 0., 1.    , 1.   )]
    ],
    basis = 'sto-3g')
    #spin=1)

m = scf.dhf.DHF(mol)
m.run()(conv_tol=1e-14)

norb = m.mo_coeff.shape[1] # norb = 6
# rotating into mo basis for stability; calling reduce instead of nested np.dots
h1e = reduce(np.dot, (m.mo_coeff.conj().T, m.get_hcore(), m.mo_coeff))
print(f'shape of h1e: {h1e.shape}')

#print(f'initial shape of 2 electron integrals \n {m._eri.shape}')
# turning a (21) shape array into a (21, 21) shape for alpha/beta off-diagonal
# (necessary because of alpha/beta degeneracy)
#eri = ao2mo.restore(4, ao2mo.general(m._eri, (mo_a, mo_a, mo_b, mo_b)), norb)
#print(f'final shape of 2 electron integrals \n {eri.shape}')
# reflecting alpha-beta across the diagonal
#eri = eri + eri.transpose(1, 0)
# transforming the alpha-alpha and beta-beta blocks 
#eri += ao2mo.restore(4, ao2mo.full(m._eri, mo_a), norb)
#eri += ao2mo.restore(4, ao2mo.full(m._eri, mo_b), norb)
# switching to no symmetry in order to go from (21, 21) to shape (6, 6, 6, 6)
eri = ao2mo.restore(1, m._eri, norb)
#print(f'final shape of 2 electron integrals \n {eri.shape}')

nelec = mol.nelectron #- 1

cisolver = fci.direct_nosym.FCI(m)
e, fcivec = cisolver.kernel()
print(f'energy: {e}')
print(f'CI coefficients: \ {fcivec}')



#e1, ci = fci.direct_nosym.kernel(h1e, eri, norb, nelec)
#print(f'CI coefficients: \n {ci}')
#rdm = fci.fci_dhf_slow.make_rdm1(ci, norb, nelec)
##print(f'Density Matrix: \n {rdm}')
#hdiag = fci.fci_dhf_slow.make_hdiag(h1e, eri, norb, nelec)
#print(f'Hamiltonian: \n {hdiag}')
#print(f'energy: {e1}')
