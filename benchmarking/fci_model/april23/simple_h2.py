import pyscf
from scipy import linalg
import torch

mol = pyscf.M(
            atom = 'H 0 0 0; H 0 0 1.1',  # in Angstrom
            basis = 'sto3g',
            symmetry = True,
            )
myhf = mol.RHF().run()


### string indexing
strings = pyscf.fci.cistring.make_strings((0,1,2,3,4,5),3)
print([bin(x) for x in strings])
print(pyscf.fci.cistring.str2addr(6, 3, 0b011001))

#
# create an FCI solver based on the SCF object
#
cisolver = pyscf.fci.FCI(myhf)
print('E(FCI) = %.12f' % cisolver.kernel()[0])
print('Coeff(RHF):')
print(cisolver.kernel()[1])

#
# create an FCI solver based on the SCF object
#
myuhf = mol.UHF().run()
cisolver = pyscf.fci.FCI(myuhf)
print('E(UHF-FCI) = %.12f' % cisolver.kernel()[0])
print('Coeff(UHF):')
print(cisolver.kernel()[1])

myghf = mol.GHF().run()
h1 = myghf.get_hcore(mol)
norb = myghf.mo_coeff.shape[1] 
eri = myghf._eri

nelec = mol.nelectron
cisolver = pyscf.fci.FCI(myghf)
print('E(GHF-FCI) = %.12f' % cisolver.kernel()[0])
print('Coeff(GHF):')
print(cisolver.kernel()[1])

mydhf = mol.DHF().set(with_gaunt=True, with_breit=True).run()
#ciolver = pyscf.fci.fci_dhf_slow.FCI(mydhf)
cisolver = pyscf.fci.FCI(mydhf)
print('E(DHF-FCI) = %.12f' % cisolver.kernel()[0])
print('Coeff(DHF):')
print(cisolver.kernel()[1])



### test to see if putting same arguments from unrestricted calc into generalized results in the same answer
h1 = myuhf.get_hcore(mol)
h1 = linalg.block_diag(h1, h1)
# norb will have to be doubled
eri = myuhf._eri
eri = pyscf.ao2mo.restore(1, eri, 3)
eri = torch.block_diag(torch.from_numpy(eri), torch.from_numpy(eri))
eri = eri.cpu().detach().numpy()
print(eri.shape)
cisolver = pyscf.fci.fci_dhf_slow.FCI(myuhf)
print('NEW E(GHF-FCI) = %.12f' % cisolver.kernel(h1, eri, norb, nelec)[0])
print('NEW Coeff(GHF):')
print(cisolver.kernel(h1, eri, norb, nelec)[1])


#
# create an FCI solver based on the given orbitals and the num. electrons and
# spin of the mol object
#
#cisolver = pyscf.fci.FCI(mol, myhf.mo_coeff)
#print('E(FCI) = %.12f' % cisolver.kernel()[0])
