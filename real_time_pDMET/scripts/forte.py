import numpy as np
from forte2 import State, MOSpace
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.ci.ci import _CIBase
from forte2 import CISigmaBuilder, RelCISigmaBuilder


class forte:
    #####################################################################

    def __init__(self, Nele):
        self.Nele = Nele

    #####################################################################

    def FCI_GS_forte(self, h, V, Ecore, Norbs, gen=False):
        # Subroutine to perform groundstate FCI calculation using Forte2
        # switching to physicist notation for the Forte2 calculation
        if gen:
            two_comp = True
            ints = SpinorbitalIntegrals.__new__(SpinorbitalIntegrals)
        else:
            two_comp = False
            ints = RestrictedMOIntegrals.__new__(RestrictedMOIntegrals)
        V_forte = np.einsum("pqrs->prqs", V).copy()

        ints.E = Ecore
        ints.H = h
        ints.V = V_forte
        mo_space = MOSpace(nmo=Norbs, active_orbitals=list(range(Norbs)))
        print("Currently calculating a singlet")
        state = State(nel=self.Nele, multiplicity=1, ms=0.0)
        if gen:
            self.ci = _CIBase(
                mo_space=mo_space,
                state=state,
                ints=ints,
                nroot=1,  # ground state
                active_orbsym=[[0] * Norbs],  # C1 symmetry
                log_level=0,
                maxiter=200,
                two_component=two_comp,  # tells CI this is spinor
                ci_algorithm="hz",  # currently default in forte2; could change
            )
        if not gen:
            self.ci = _CIBase(
                mo_space=mo_space,
                state=state,
                ints=ints,
                nroot=1,  # ground state
                active_orbsym=[[0] * Norbs],  # C1 symmetry
                log_level=0,
                maxiter=200,
                ci_algorithm="hz",  # currently default in forte2; could change
            )

        self.ci.run()

        CIcoeffs = self.ci.evecs
        E_FCI = self.ci.E[0]

        return CIcoeffs, E_FCI

    #####################################################################

    def setup_ci(self, h, V, Ecore, Norbs, gen=False):
        if gen:
            two_comp = True
            ints = SpinorbitalIntegrals.__new__(SpinorbitalIntegrals)
        else:
            two_comp = False
            ints = RestrictedMOIntegrals.__new__(RestrictedMOIntegrals)

        V_forte = np.einsum("pqrs->prqs", V).copy()

        ints.E = Ecore
        ints.H = h
        ints.V = V_forte

        mo_space = MOSpace(nmo=Norbs, active_orbitals=list(range(Norbs)))
        state = State(nel=self.Nele, multiplicity=1, ms=0.0)

        if gen:
            self.ci = _CIBase(
                mo_space=mo_space,
                state=state,
                ints=ints,
                nroot=1,
                active_orbsym=[[0] * Norbs],
                log_level=0,
                maxiter=200,
                two_component=two_comp,
                ci_algorithm="hz",
            )
        if not gen:
            self.ci = _CIBase(
                mo_space=mo_space,
                state=state,
                ints=ints,
                nroot=1,
                active_orbsym=[[0] * Norbs],
                log_level=0,
                maxiter=200,
                two_component=two_comp,
                ci_algorithm="hz",
            )

        # Replicate just the sigma builder setup portion of run()
        # without triggering the Davidson-Liu diagonalization
        self.ci._ci_solver_startup()
        if gen:
            self.ci.ci_sigma_builder = RelCISigmaBuilder(
                self.ci.ci_strings,
                self.ci.ints.E.real,
                self.ci.ints.H,
                self.ci.ints.V,
                self.ci.log_level,
            )
        else:
            self.ci.ci_sigma_builder = CISigmaBuilder(
                self.ci.ci_strings,
                self.ci.ints.E,
                self.ci.ints.H,
                self.ci.ints.V,
                self.ci.log_level,
            )
        self.ci.ci_sigma_builder.set_memory(self.ci.ci_builder_memory)
        self.ci.ci_sigma_builder.set_algorithm("hz")

        # After creating ci_sigma_builder, allocate sigma_det and b_det
        # to match what run() would have set up via _ci_solver_startup()
        if gen:
            self.ci.sigma_det = np.zeros((self.ci.ndet,), dtype=complex)
            self.ci.b_det = np.zeros((self.ci.ndet,), dtype=complex)
        else:
            self.ci.sigma_det = np.zeros((self.ci.ndet,))
            self.ci.b_det = np.zeros((self.ci.ndet,))

    #####################################################################

    def applyham_forte2(self, CIcoeffs):
        CIcoeffs_out = np.zeros_like(CIcoeffs)
        b_det = CIcoeffs.flatten().astype(np.complex128)

        # Compute the sigma block from the basis block; assume single state CIcoeffs
        # copies ensure contiguous arrays are passed to C++
        # Hamiltonian is really apply hamiltonian; H and V hidden in ci class
        self.ci.ci_sigma_builder.Hamiltonian(b_det, self.ci.sigma_det)
        # reshape output back to match input shape
        CIcoeffs_out = self.ci.sigma_det.copy().reshape(CIcoeffs.shape)
        return CIcoeffs_out

    #####################################################################

    def get_corr1RDM(self, CIcoeffs, gen=False):
        # Subroutine to get the FCI 1RDM
        sigma_builder = self.ci.ci_sigma_builder
        if gen:
            CIcoeffs = CIcoeffs.flatten().astype(np.complex128)
            corr1RDM = sigma_builder.so_1rdm(CIcoeffs.copy(), CIcoeffs.copy())
            corr1RDM = np.transpose(corr1RDM)
        else:
            corr1RDM = self.ci.make_sf_1rdm(0)
        return corr1RDM

    #####################################################################

    def get_corr12RDM(self, CIcoeffs, gen=False):
        # Subroutine to get the FCI 1 & 2 RDMs together

        # need a CI_basetype object for forte2
        sigma_builder = self.ci.ci_sigma_builder
        if gen:
            CIcoeffs = CIcoeffs.flatten().astype(np.complex128)
            corr1RDM = sigma_builder.so_1rdm(CIcoeffs.copy(), CIcoeffs.copy())
            corr1RDM = np.transpose(corr1RDM)
            corr2RDM = sigma_builder.so_2rdm(CIcoeffs.copy(), CIcoeffs.copy())
        else:
            corr1RDM = self.ci.make_sf_1rdm(0)
            corr2RDM = self.ci.make_sf_2rdm(0)
        return corr1RDM, corr2RDM

    #####################################################################
