import numpy as np
import real_time_pDMET.scripts.utils as utils
import pyscf.fci
from pyscf import gto, scf, ao2mo
import sys
import scipy.linalg as la


###########################################################


def FCI_GS(h, V, U, Norbs, Nele, gen=False):
    if isinstance(Nele, tuple):
        Nele = sum(Nele)
    # Define PySCF molecule
    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True
    if Nele // 2:
        mol.spin = 0
    else:
        mol.spin = 1

    if not gen:
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h

        mf.get_ovlp = lambda *args: np.eye(Norbs)

        mf._eri = ao2mo.restore(8, V, Norbs)
        # taking advantage of symmetry in 2e term
        # (assuming orbitals are real - 8fold symmetry)
        # if orbitals are complex - 4 fold symmetry
        mf.kernel()
        sys.stdout.flush()

        # FCI calculation using HF molecular orbitals

        # might be useful to use direct_uhf.FCI() instead for the cisolver
        # Second - FCI calculation using HF molecular orbitals
        cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
        E_FCI, CIcoeffs = cisolver.kernel()

        # Need to rotate CI coefficients back to embeding basis
        # used in DMET (because now they are in orbital basis)

        CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
            CIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff)
        )

    if gen:
        E_FCI, CIcoeffs = pyscf.fci.fci_dhf_slow.kernel(h, V, Norbs, Nele)

        # h = utils.reshape_gtor_matrix(h)
        # V = utils.reshape_gtor_tensor(V)

        # E_FCI, CIcoeffs = pyscf.fci.fci_dhf_slow.kernel(h, V, Norbs, Nele)

    return CIcoeffs, E_FCI


###########################################################


def get_corr1RDM(CIcoeffs, Norbs, Nele, gen=False):
    # subroutine to FCI 1RDM

    if not gen:
        # restricted notation is dm_pq = < q^+ p >
        # C = RC +i IC => can rewrite:
        # PySCF uses only density matricies for real numbers,
        # so procedure is split into complex/real parts
        # <psi|a+a|psi> = <Rpsi|~|Rpsi> + <Ipsi|~|Ipsi>
        # + i<Rpsi|~|Ipsi> - i<Ipsi|~|Rpsi>
        # transition density matrix  = any <a | ~ | b>

        if np.iscomplexobj(CIcoeffs):
            Re_CIcoeffs = np.copy(CIcoeffs.real)
            Im_CIcoeffs = np.copy(CIcoeffs.imag)

            corr1RDM = 1j * pyscf.fci.direct_spin1.trans_rdm1(
                Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele
            )

            corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1(
                Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele
            )

            corr1RDM += pyscf.fci.direct_spin1.make_rdm1(Re_CIcoeffs, Norbs, Nele)
            corr1RDM += pyscf.fci.direct_spin1.make_rdm1(Im_CIcoeffs, Norbs, Nele)
        else:
            corr1RDM = pyscf.fci.direct_spin1.make_rdm1(CIcoeffs, Norbs, Nele)

    if gen:
        # (notes from dynamics:)
        # Notation for generalized 1RDM from fci_dhf_slow is dm_pq = <|p^+ q|>
        # PySCF requires CIcoeffs to be in a spin-blocked configuration
        corr1RDM = pyscf.fci.fci_dhf_slow.make_rdm1(CIcoeffs, Norbs, Nele)

        if not np.allclose(np.diag(corr1RDM.imag), 0, atol=1e-9):
            print(
                "WARNING: NON-NEGLIGIBLE COMPLEX TERMS ALONG DIAGONAL OF EMBEDDED CORRELATED 1RDM"
            )
            print("-------- ENDING SIMULATION --------")
            exit()

        np.fill_diagonal(
            corr1RDM, corr1RDM.diagonal().real
        )  # make diagonal elements real

        # tranpose back to dm_pq = <|q^+ p|> to match restricted case
        corr1RDM = np.transpose(corr1RDM)

        if not la.ishermitian(corr1RDM, atol=1e-9):
            print("WARNING: EMBEDDED CORRELATED 1RDM IS NOT HERMITIAN")
            print("-------- ENDING SIMULATION --------")
            exit()

        corr1RDM = utils.make_hermitian(corr1RDM)

    return corr1RDM


###########################################################


def get_corr12RDM(CIcoeffs, Norbs, Nele, gen=False):
    # Subroutine to get the FCI 1 & 2 RDMs together

    if not gen:
        # Notation for 1RDM is rho_pq = < c_q^dag c_p >
        # Notation for 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

        if np.iscomplexobj(CIcoeffs):
            Re_CIcoeffs = np.copy(CIcoeffs.real)
            Im_CIcoeffs = np.copy(CIcoeffs.imag)

            corr1RDM, corr2RDM = pyscf.fci.direct_spin1.trans_rdm12(
                Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele
            )

            corr1RDM = corr1RDM * 1j
            corr2RDM = corr2RDM * 1j

            tmp1, tmp2 = pyscf.fci.direct_spin1.trans_rdm12(
                Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele
            )

            corr1RDM -= 1j * tmp1
            corr2RDM -= 1j * tmp2

            tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12(Re_CIcoeffs, Norbs, Nele)

            corr1RDM += tmp1
            corr2RDM += tmp2

            tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12(Im_CIcoeffs, Norbs, Nele)

            corr1RDM += tmp1
            corr2RDM += tmp2

        else:
            corr1RDM, corr2RDM = pyscf.fci.direct_spin1.make_rdm12(
                CIcoeffs, Norbs, Nele
            )

    if gen:
        # Notation for generalized 1RDM is dm_pq = <|p^+ q|>
        # Notation for generalized 2RDM is dm_pq,rs = <|p^+ q r^+ s|>
        # This would be equivalent to (p_dag r_dag s q) in chemists notation, so equal to restricted notation
        # PySCF requires CIcoeffs to be in a spin-blocked configuration

        corr1RDM, corr2RDM = pyscf.fci.fci_dhf_slow.make_rdm12(CIcoeffs, Norbs, Nele)

        if not np.isclose(la.norm(CIcoeffs), 1.0, atol=1e-3):
            print(f"norm of CIcoeffs: {la.norm(CIcoeffs)}")

        if not np.allclose(np.diag(corr1RDM.imag), 0, atol=1e-9):
            print(la.norm(CIcoeffs))
            print(
                "WARNING: NON-NEGLIGIBLE COMPLEX TERMS ALONG DIAGONAL OF EMBEDDED CORRELATED 1RDM"
            )
            print("-------- ENDING SIMULATION --------")
            exit()

        np.fill_diagonal(
            corr1RDM, corr1RDM.diagonal().real
        )  # make diagonal elements real
        corr1RDM = np.transpose(corr1RDM)

        if not la.ishermitian(corr1RDM, atol=1e-9):
            print("WARNING: EMBEDDED CORRELATED 1RDM IS NOT HERMITIAN")
            print("-------- ENDING SIMULATION --------")
            exit()

        corr1RDM = utils.make_hermitian(corr1RDM)

    return corr1RDM, corr2RDM
