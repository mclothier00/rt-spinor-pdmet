import numpy as np
import scipy.linalg as la
from real_time_pDMET.scripts.utils import diagonalize
from pyscf import gto, scf, ao2mo

#####################################################################


def interactive_RHF(Nele, h_site, V_site):
    Norbs = Nele
    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h_site
    mf.get_ovlp = lambda *args: np.eye(Norbs)
    mf._eri = ao2mo.restore(8, V_site, Norbs)
    mf.kernel()
    mf1RDM = mf.make_rdm1()

    return mf1RDM
#####################################################################


## NOTE: HAVE NOT TESTED
def interactive_GHF(Nele, h_site, V_site):
    Norbs = Nele*2
    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True
    mf = scf.GHF(mol)
    mf.get_hcore = lambda *args: h_site
    mf.get_ovlp = lambda *args: np.eye(Norbs)
    mf._eri = ao2mo.restore(8, V_site, Norbs)
    mf.kernel()
    mf1RDM = mf.make_rdm1()

    return mf1RDM
#####################################################################


def rhf_calc_hubbard(Nelec, Hcore):

    # simplified subroutine to perform a
    # mean-field (ie U=0) calculation for Hubbard model

    # Diagonalize hopping-hamiltonian
    evals, orbs = diagonalize(Hcore)
    # Form the 1RDM
    P = rdm_1el(orbs, int(Nelec/2))
    return P
#####################################################################


## NOTE: HAVE NOT TESTED
def ghf_calc_hubbard(Nelec, Hcore):

    # simplified subroutine to perform a
    # mean-field (ie U=0) calculation for Hubbard model

    # Diagonalize hopping-hamiltonian
    evals, orbs = diagonalize(Hcore)
    # Form the 1RDM
    P = rdm_1el(orbs, int(Nelec))
    return P
#####################################################################


## NOTE: will need to change this to rdm_1el_rhf at end
def rdm_1el(C, Ne):
    # subroutine that calculates and
    # returns the one-electron density matrix in original site basis
    Cocc = C[:, :Ne]
    P = 2*np.dot(Cocc, np.transpose(np.conjugate(Cocc)))
    return P
#####################################################################


## NOTE: HAVE NOT TESTED; is Ne changed in the generalized formalism 
## to account for larger dimension?
def rdm_1el_ghf(C, Ne):
    # subroutine that calculates and
    # returns the one-electron density matrix in original site basis
    Cocc = C[:, :Ne]
    P = np.dot(Cocc, np.transpose(np.conjugate(Cocc)))
    return P
#####################################################################