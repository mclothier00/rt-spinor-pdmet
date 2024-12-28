# Mod that contains subroutines associated with pyscf FCI calculations

import numpy as np
import real_time_pDMET.scripts.utils as utils
import real_time_pDMET.scripts.applyham_pyscf as applyham_pyscf
import pyscf.fci
from pyscf import gto, scf, ao2mo
#####################################################################


def FCI_GS(h, V, Ecore, Norbs, Nele, gen=False):
    # Subroutine to perform groundstate FCI calculation using pyscf

    if isinstance(Nele, tuple):
        Nele = sum(Nele)

    # Define pyscf molecule
    mol = gto.M()
    mol.nelectron = Nele
    mol.nao = Norbs
    # this call is necessary to use user defined hamiltonian in fci step
    mol.incore_anyway = True
    if Nele // 2:
        mol.spin = 0
    else:
        mol.spin = 1

    if not gen:
        print(
            "WARNING: Currently GS FCI coefficients are not in final embedding basis."
        )

        # First perform restricted HF calculation
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h
        mf.get_ovlp = lambda *args: np.eye(Norbs)
        mf._eri = ao2mo.restore(8, V, Norbs)
        mf.kernel()

        # Perform FCI calculation using HF MOs
        cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
        E_FCI, CIcoeffs = cisolver.kernel()
        # E_FCI, CIcoeffs = pyscf.fci.direct_spin1.kernel(h, V, Norbs, Nele)
        # print(f"fci energy: {E_FCI}")
        # print(f"CIcoeffs before transformation: {CIcoeffs}")

        # NOTE: currently commenting out these lines for TDFCI; put back in
        #       for DMET

        CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
            CIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff)
        )
        print(f"CIcoeffs after transformation: {CIcoeffs}")

    if gen:
        # NOTE: HF det should be the most dominant det in FCI exp.,
        #       making num solver more stable than without.
        #       may not get correct answer for lattice models, so check
        #       for proper convergence.

        # First perform generalized HF calculation
        # mf = scf.GHF(mol)
        # mf.get_hcore = lambda *args: h
        # mf.get_ovlp = lambda *args: np.eye(Norbs)
        # mf._eri = ao2mo.restore(1, V, Norbs)
        # mf.kernel()
        # V_nosymm = ao2mo.restore(1, V, Norbs)
        # print(mf.mo_coeff)
        # Perform FCI calculation using HF MOs
        # cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
        # cisolver = pyscf.fci.fci_dhf_slow(mf)  # , mf.mo_coeff)
        E_FCI, CIcoeffs = pyscf.fci.fci_dhf_slow.kernel(h, V, Norbs, Nele)
        print(CIcoeffs)
        # print(f"fci energy: {E_FCI}")
        # Rotate CI coefficients back to basis used in DMET calculations
        # print(f"test: {np.dot(utils.adjoint(mf.mo_coeff),CIcoeffs)}")
        # CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
        #    CIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff)
        # )

    return CIcoeffs


#####################################################################

# NOTE: still need to check that indexing is correct


def get_corr1RDM(CIcoeffs, Norbs, Nele, gen=False):
    # Subroutine to get the FCI 1RDM
    # notation for restricted is rho_pq = < c_q^dag c_p >

    if not gen:
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

    # notation for generalized is rho_pq = <|p^+ q|> in code but
    # rho_pq = <|q^+ p|> in docs... CHECK!!

    if gen:
        corr1RDM = pyscf.fci.fci_dhf_slow.make_rdm1(CIcoeffs, Norbs, Nele)

    return corr1RDM


#####################################################################


def get_corr12RDM(CIcoeffs, Norbs, Nele, gen=False):
    # Subroutine to get the FCI 1 & 2 RDMs together
    # Notation for restricted 1RDM is rho_pq = < c_q^dag c_p >
    # Notation for restricted 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

    if not gen:
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
            corr1RDM = pyscf.fci.direct_spin1.make_rdm1(CIcoeffs, Norbs, Nele)
    # Notation for generalized 1RDM is dm_pq = <|p^+ q|>
    # Notation for generalized 2RDM is dm_pq,rs = <|p^+ q r^+ s|>
    # This would be equivalent to (p_dag r_dag s q) in chemists notation, so equal to restricted notation
    # PySCF requires CIcoeffs to be in a spin-blocked configuration
    if gen:
        # Nele = 2
        corr1RDM, corr2RDM = pyscf.fci.fci_dhf_slow.make_rdm12(CIcoeffs, Norbs, Nele)

    return corr1RDM, corr2RDM


#####################################################################

# not changed


def get_trans1RDM(CIcoeffs_1, CIcoeffs_2, Norbs, Nele):
    # Subroutine to get the transition 1RDM between two CI vectors
    # notation is rho_pq = < psi_1 | c_q^dag c_p | psi_2 >

    if np.iscomplexobj(CIcoeffs_1) or np.iscomplexobj(CIcoeffs_2):
        Re_CIcoeffs_1 = np.copy(CIcoeffs_1.real)
        Im_CIcoeffs_1 = np.copy(CIcoeffs_1.imag)

        Re_CIcoeffs_2 = np.copy(CIcoeffs_2.real)
        Im_CIcoeffs_2 = np.copy(CIcoeffs_2.imag)

        corr1RDM = 1j * pyscf.fci.direct_spin1.trans_rdm1(
            Re_CIcoeffs_1, Im_CIcoeffs_2, Norbs, Nele
        )

        corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1(
            Im_CIcoeffs_1, Re_CIcoeffs_2, Norbs, Nele
        )

        corr1RDM += pyscf.fci.direct_spin1.trans_rdm1(
            Re_CIcoeffs_1, Re_CIcoeffs_2, Norbs, Nele
        )

        corr1RDM += pyscf.fci.direct_spin1.trans_rdm1(
            Im_CIcoeffs_1, Im_CIcoeffs_2, Norbs, Nele
        )

    else:
        corr1RDM = pyscf.fci.direct_spin1.trans_rdm1(
            CIcoeffs_1, CIcoeffs_2, Norbs, Nele
        )

    return corr1RDM


#####################################################################


def get_FCI_E(h, V, Econst, CIcoeffs, Norbs, Nalpha, Nbeta, gen=False):
    # Subroutine to calculate the FCI electronic energy
    # for given Hamiltonian and FCI vector
    # Works with complex Hamitlonian and FCI vector

    if not gen:
        Hpsi = applyham_pyscf.apply_ham_pyscf_fully_complex(
            CIcoeffs, h, V, Nalpha, Nbeta, Norbs, Econst
        )

        Re_Hpsi = np.copy(Hpsi.real)
        Im_Hpsi = np.copy(Hpsi.imag)

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        FCI_E = pyscf.fci.addons.overlap(Re_CIcoeffs, Re_Hpsi, Norbs, (Nalpha, Nbeta))
        FCI_E += pyscf.fci.addons.overlap(Im_CIcoeffs, Im_Hpsi, Norbs, (Nalpha, Nbeta))
        FCI_E += 1j * pyscf.fci.addons.overlap(
            Re_CIcoeffs, Im_Hpsi, Norbs, (Nalpha, Nbeta)
        )
        FCI_E -= 1j * pyscf.fci.addons.overlap(
            Im_CIcoeffs, Re_Hpsi, Norbs, (Nalpha, Nbeta)
        )
        # print(f"original: {FCI_E}")

        h2e = pyscf.fci.direct_nosym.absorb_h1e(h, V, Norbs, (Nalpha + Nbeta), 0.5)
        ci1 = pyscf.fci.direct_nosym.contract_2e(h2e, CIcoeffs, Norbs, (Nalpha + Nbeta))
        # print(f"ci1 from direct_nosym: \n {h2e}")
        #        print(f"from documentation: {np.dot(CIcoeffs.reshape(-1), ci1.reshape(-1))}")
        # from apply_hams call used above
        h2e = pyscf.fci.direct_spin1.absorb_h1e(h, V, Norbs, (Nalpha, Nbeta), 0.5)
        ci1 = pyscf.fci.direct_spin1.contract_2e(h2e, CIcoeffs, Norbs, (Nalpha, Nbeta))
        # print(f"ci1 from direct_spin1: \n {h2e}")

        FCI_E = pyscf.fci.direct_nosym.energy(h, V, CIcoeffs, Norbs, (Nalpha + Nbeta))
        #        print(f"from energy call directly: {FCI_E}")

        FCI_E = pyscf.fci.direct_spin1.kernel(h, V, Norbs, (Nalpha, Nbeta))[0]
        # print(f"from kernel: {FCI_E}")

    if gen:
        Nelec = Nalpha + Nbeta
        Hpsi = applyham_pyscf.apply_ham_pyscf_spinor(
            CIcoeffs, h, V, Nelec, Norbs, Econst
        )

        # NOTE: only difference between energy call and overlap procedure is in energy call ci1 is not
        # added to the original CIcoeffs while in Hpsi spinor

        Re_Hpsi = np.copy(Hpsi.real)
        Im_Hpsi = np.copy(Hpsi.imag)

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        FCI_E = pyscf.fci.addons.overlap(Re_CIcoeffs, Re_Hpsi, Norbs, Nelec)
        FCI_E += pyscf.fci.addons.overlap(Im_CIcoeffs, Im_Hpsi, Norbs, Nelec)
        FCI_E += 1j * pyscf.fci.addons.overlap(Re_CIcoeffs, Im_Hpsi, Norbs, Nelec)
        FCI_E -= 1j * pyscf.fci.addons.overlap(Im_CIcoeffs, Re_Hpsi, Norbs, Nelec)
        # print(f"original: {FCI_E}")

        # did not match
        # taken from pyscf's energy call in fci_dhf_slow documentation
        h2e = pyscf.fci.fci_dhf_slow.absorb_h1e(h, V, Norbs, Nelec, 1.0)
        ci1 = pyscf.fci.fci_dhf_slow.contract_2e(h2e, CIcoeffs, Norbs, Nelec)
        FCI_E = np.dot(CIcoeffs.conj(), ci1)
        #        print(f"from documentation: {FCI_E}")
        # print(f"ci1 from dhf_slow: \n {h2e}")

        FCI_E, CI_coeffsnew = pyscf.fci.fci_dhf_slow.kernel(h, V, Norbs, Nelec)
        # print(f"from kernel: {FCI_E}")
        # print(
        #    f"difference between old and new CI coefficients: \n {CI_coeffsnew - CIcoeffs}"
        # )

        # FCI_E = pyscf.fci.fci_dhf_slow.kernel_dhf(CIcoeffs, h, V, Norbs, Nelec)[0]
        # print(f"from kernel_dhf: {FCI_E}")

    return FCI_E.real


#####################################################################
