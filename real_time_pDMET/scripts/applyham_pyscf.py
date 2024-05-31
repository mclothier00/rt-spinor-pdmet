# PYTHON CODE FROM PYSCF FOR APPLYING THE HAMILTONIAN TO A FCI
# VECTOR WITHOUT ASSUMING ANY SYMMETRY OF THE HAMILTONIAN
# ONLY KEPT NECESSARY SUBROUTINES:
# ORIGNAL FILE OBTAINED FROM PYSCF FILE direct_spin0_o0.py
# contract_2e_complex WAS OBTAINED FROM PYSCF FILE direct_spin0_o0.py,
# REMOVING CALLS TO pyscf.ao2mo.restore
# absorb_1e WAS OBTAINED FROM PYSCF FILE direct_spin1.py,
# REMOVING CALLS TO pyscf.ao2mo.restore

import numpy
import pyscf.lib
import pyscf.ao2mo
import pyscf.fci
from pyscf.fci import cistring

#####################################################################

## NOTE: need to add additional function to check dimensions and sned to spinor
## if in spinor basis?
def apply_ham_pyscf_check(
        CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    '''
     subroutine that checks if the hamiltonian is real or complex
     and then calls the appropriate subroutine to apply the
     hamiltonian to a vector of CI coefficients using pyscf
    '''
    if(numpy.iscomplexobj(hmat) and numpy.iscomplexobj(Vmat)):

        # Complex hamiltonian
        CIcoeffs = apply_ham_pyscf_complex(
            CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr)

    elif(not numpy.iscomplexobj(hmat) and not numpy.iscomplexobj(Vmat)):

        # Real hamiltonian
        CIcoeffs = apply_ham_pyscf_real(
            CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr)

    else:

        print('ERROR: the 1e- integrals and 2e- integrals in')
        print('applyham_pyscf.apply_ham_pyscf_check')
        print('are NOT both real nor both complex')
        exit()
    return CIcoeffs
#####################################################################

## NOTE: needs to take in a 1d fci array, not 2d.

def apply_ham_pyscf_spinor(
        CIcoeffs, hmat, Vmat, nelec, norbs, Econst, fctr=1.0):
    '''
    NOTE: This subroutine calls the PySCF fci solver for DHF or GHF,
     which can handle a complex Hamiltonian in a spinor basis. However, 
     both hmat and Vmat must be either both complex or both real. This 
     subroutine calls PySCF to apply a hamiltonian to a vector
     of CI coefficients.
     CIcoeffs is a 1d-array containing the CI coefficients in which
     the rows are coefficients for each configuration/determinant and
     the strings are ordered in ascending binary order with
     a 0/1 implies that a SPINOR orbital is empty/occupied.
     Vmat is the 2e- integrals and are given in chemistry notation.
     Econst is a constant energy contribution to the Hamiltonian.
     fctr is the factor in front of the 2e- terms
     when defining the hamiltonian; because this is a spinor basis, 
     this is set to 1.0.
     NOTE: Currently only accepts hamiltonians and norbs in dimensionality 
     of the spin-generalized formalism. 
     '''
    
    Vmat = pyscf.fci.fci_dhf_slow.absorb_h1e(
        hmat, Vmat, norbs, nelec, fctr)

    # this errors out as it wants a single vector, not a matrix or list of vectors
    # april 24 NOTE: this should just work now...
    temp = pyscf.fci.fci_dhf_slow.contract_2e(
        Vmat, CIcoeffs, norbs, nelec)

    CIcoeffs = temp + Econst*CIcoeffs

#    row, col = CIcoeffs.shape
#    temp = numpy.zeros((row, col))
#    print(f'original CI coefficient matrix: \n {CIcoeffs}')
#    print(f'CI vector: {CIcoeffs[:,0]}')
#    
#    for i in range(0, row):
#        CI_vec = CIcoeffs[i, :]
#        temp[i, :] = pyscf.fci.fci_dhf_slow.contract_2e(
#            Vmat, CI_vec, norbs, nelec)
#    print(f'shape of temp: {temp.shape}')
#    CIcoeffs = temp + Econst*CIcoeffs

    return CIcoeffs
#####################################################################

def apply_ham_pyscf_fully_complex(
        CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    '''
     subroutine that uses the apply_ham_pyscf_nosym
     subroutine below to apply a complex hamiltonian
     to a complex set of CI coefficients -
     also works if some subset are real, it's just slower
    '''
    CIcoeffs = (apply_ham_pyscf_nosym(
        numpy.copy(CIcoeffs.real), numpy.copy(hmat.real),
        numpy.copy(Vmat.real), nalpha, nbeta, norbs, Econst, fctr)
                - apply_ham_pyscf_nosym(
                    numpy.copy(CIcoeffs.imag), numpy.copy(hmat.imag),
                    numpy.copy(Vmat.imag), nalpha, nbeta, norbs, 0.0, fctr)
                + 1j*(apply_ham_pyscf_nosym(
                    numpy.copy(CIcoeffs.imag), numpy.copy(hmat.real),
                    numpy.copy(Vmat.real), nalpha, nbeta, norbs, Econst, fctr)
                      + apply_ham_pyscf_nosym(
                          numpy.copy(CIcoeffs.real), numpy.copy(hmat.imag),
                          numpy.copy(Vmat.imag), nalpha, nbeta,
                          norbs, 0.0, fctr)))
    return CIcoeffs
#####################################################################


def apply_ham_pyscf_real(
        CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    '''
    NOTE: THIS SUBROUTINE ASSUMES THAT THE
     HAMILTONIAN IS SYMMETRIC (HERMITIAN AND REAL)
     AND IS CALLING PYSCF TO APPLY THE HAMILTONIAN
     subroutine to apply a hamiltonian to a vector
     of CI coefficients using pyscf
     CIcoeffs is a 2d-array containing the CI coefficients,
     the rows/columns correspond to the alpha/beta strings
     the strings are ordered in asscending binary order with
     a 0/1 implies that an orbital is empty/occupied
     the 2e- integrals, Vmat, are given in chemistry notation
     Econst is a constant energy contribution to the hamiltonian
     fctr is the factor in front of the 2e- terms
     when defining the hamiltonian
     '''
    Vmat = pyscf.fci.direct_spin1.absorb_h1e(
        hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    temp = pyscf.fci.direct_spin1.contract_2e(
        Vmat, CIcoeffs, norbs, (nalpha, nbeta))
    CIcoeffs = temp + Econst*CIcoeffs

    return CIcoeffs
#####################################################################

# NOTE: this is edited for print statements; switch back to normal after testing

def apply_ham_pyscf_nosym(
        CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    '''
    NOTE: THIS SUBROUTINE MAKES NO ASSUMPTION ABOUT THE SYMMETRY OF
    THE HAMILTONIAN, BUT CI COEFFICIENTS AND HAMILTONIAN MUST BE REAL
    AND IS CALLING PYSCF TO APPLY THE HAMILTONIAN
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
    print(f'original CI coefficients: {CIcoeffs.shape}')
    
    Vmat_new = pyscf.fci.direct_nosym.absorb_h1e(
        hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    temp = pyscf.fci.direct_nosym.contract_2e(
        Vmat_new, CIcoeffs, norbs, (nalpha, nbeta))
    CIcoeffs_new = temp + Econst*CIcoeffs

    print(f'CI coefficients after applyhams: \n {CIcoeffs_new}')
    nelec = nalpha + nbeta

    CIgeneral = apply_ham_pyscf_spinor(
        CIcoeffs, hmat, Vmat, nelec, norbs, Econst, fctr=1.0)
    print(f'CI coefficients after general applyhams: \n {CIgeneral}')

    exit()
    return CIcoeffs_new
#####################################################################


def apply_ham_pyscf_complex(
        CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    '''
    NOTE: THIS SUBROUTINE ALLOWS FOR COMPLEX HAMILTONIAN,
    BUT ONLY REAL CI COEFFICIENTS
    AND IS USING THE SUBROUTINES IN THIS MODULE TO APPLY THE HAMILTONIAN
    subroutine to apply a hamiltonian to a vector
    of CI coefficients using pyscf
    CIcoeffs is a 2d-array containing the CI coefficients,
    the rows/columns correspond to the alpha/beta strings
    the strings are ordered in asscending binary order with a
    0/1 implies that an orbital is empty/occupied
    the 2e- integrals, Vmat, are given in chemistry notation
    Econst is a constant energy contribution to the hamiltonian
    fctr is the factor in front of the 2e- terms when defining the hamiltonian
    '''
    Vmat = absorb_h1e_complex(
        hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    temp = contract_2e_complex(
        Vmat, CIcoeffs, norbs, (nalpha, nbeta))
    CIcoeffs = temp + Econst*CIcoeffs

    return CIcoeffs
#####################################################################


def contract_2e_complex(g2e, fcivec, norb, nelec, link_index=None):
    '''
    version of the pyscf subroutine contract_2e
    which allows for complex orbitals
    still assumes real CI coefficients
    removed calls to pyscf.ao2mo.restore
    other changes from pyscf have been noted
    subroutine follows logic of
    eqs 11.8.13-11.8.15 in helgaker, jorgensen and olsen
    '''

    neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    fcivec = fcivec.reshape(na, nb)

    t1 = numpy.zeros((norb, norb, na, nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * fcivec[str0]
    for k in range(na):
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[a, i, k, str1] += sign * fcivec[k, str0]

    # following line assumes the symmetry
    # that g[p,q,r,s]=g[r,s,p,q] in chemists notation
    # this symmetry holds for real and complex orbitals
    t1 = numpy.dot(g2e.reshape(norb*norb, -1), t1.reshape(norb*norb, -1))
    t1 = t1.reshape(norb, norb, na, nb)

    # data type of ci1 is now complex
    ci1 = numpy.zeros_like(fcivec, dtype=complex)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # indices a and i have been switched from pyscf
            ci1[str0] += sign * t1[i, a, str1]
    for k in range(na):
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                # indices a and i have been switched from pyscf
                ci1[k, str0] += sign * t1[i, a, k, str1]

    return ci1
#####################################################################


def absorb_h1e_complex(h1e, eri, norb, nelec, fac=1):
    # absorbing 1e- terms into 2e- terms
    # following 11.8.8 in helgaker, jorgensen and olsen
    # allows for no assumption about symmetry of 1 and 2 e- terms

    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    # eri = eri.copy()
    # h2e = pyscf.ao2mo.restore(1, eri, norb)
    h2e = eri.copy()
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./nelec)
    for k in range(norb):
        h2e[k, k, :, :] += f1e
        h2e[:, :, k, k] += f1e

    return h2e * fac
