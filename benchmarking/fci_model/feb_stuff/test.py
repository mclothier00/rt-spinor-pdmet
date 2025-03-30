import numpy
from pyscf import fci

U = 3.0
j1 = 0.8 
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
h1e = numpy.asarray(h1e)
eri = numpy.zeros((4,4,4,4), dtype=complex)

norb = 4
nelec = 2
Econst = 1
e, CIcoeffs = fci.fci_dhf_slow.kernel(h1e, eri, norb, nelec, nroots=6)
CIcoeffs = numpy.asarray(CIcoeffs)


link_index = fci.cistring.gen_linkstr_index(range(norb), nelec)
na = fci.cistring.num_strings(norb, nelec)
t1 = numpy.zeros((norb, norb, na), dtype=eri.dtype)
for str0, tab in enumerate(link_index):
    for a, i, str1, sign in tab:
        t1[a, i, str1] += sign * CIcoeffs[str0]

exit()

def apply_ham_pyscf_fully_complex(
        CIcoeffs, hmat, Vmat, nelec, norbs, Econst, fctr=0.5):
    '''
     subroutine that uses the apply_ham_pyscf_nosym
     subroutine below to apply a complex hamiltonian
     to a complex set of CI coefficients -
     also works if some subset are real, it's just slower
    '''
    CIcoeffs = (apply_ham_pyscf_dhf_slow(
        numpy.copy(CIcoeffs.real), numpy.copy(hmat.real),
        numpy.copy(Vmat.real), nelec, norbs, Econst, fctr)
                - apply_ham_pyscf_dhf_slow(
                    numpy.copy(CIcoeffs.imag), numpy.copy(hmat.imag),
                    numpy.copy(Vmat.imag), nelec, norbs, 0.0, fctr)
                + 1j*(apply_ham_pyscf_dhf_slow(
                    numpy.copy(CIcoeffs.imag), numpy.copy(hmat.real),
                    numpy.copy(Vmat.real), nelec, norbs, Econst, fctr)
                      + apply_ham_pyscf_dhf_slow(
                          numpy.copy(CIcoeffs.real), numpy.copy(hmat.imag),
                          numpy.copy(Vmat.imag), nelec,
                          norbs, 0.0, fctr)))
    return CIcoeffs

## NOTE: check that factor should be one rather than 1/2
## NOTE: CHECK IF ONE MATRIX CAN BE COMPLEX AND ONE REAL
def apply_ham_pyscf_dhf_slow(
        CIcoeffs, hmat, Vmat, nelec, norbs, Econst, fctr=1):
    '''
    NOTE: THIS SUBROUTINE MAKES NO ASSUMPTION ABOUT THE SYMMETRY OF
    THE HAMILTONIAN, IF EITHER HMAT OR VMAT IS COMPLEX, THE OTHER MUST BE COMPLEX
    subroutine to apply a hamiltonian to a vector of
    CI coefficients using pyscf
    CIcoeffs is a 2d-array containing the CI coefficients,
    the rows/columns correspond to the alpha/beta strings
    the strings are ordered in asscending binary order with
    a 0/1 implies that an orbital is empty/occupied
    the 2e- integrals, Vmat, are given in chemistry notation
    Econst is a constant energy contribution to the hamiltonian
    fctr is the factor in front of the 2e- terms when defining the hamiltonian
    '''
    Vmat = fci.fci_dhf_slow.absorb_h1e(
        hmat, Vmat, norbs, nelec, fctr)
    temp = fci.fci_dhf_slow.contract_2e(
        Vmat, CIcoeffs, norbs, nelec)
    CIcoeffs = temp + Econst*CIcoeffs

    return CIcoeffs

ci = apply_ham_pyscf_dhf_slow(CIcoeffs, h1e, eri, nelec, norb, Econst)
print(ci)
