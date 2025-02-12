# including all quantities specific to a given fragment
import numpy as np
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod
import real_time_pDMET.scripts.utils as utils
import real_time_pDMET.scripts.applyham_pyscf as applyham_pyscf
import time

# ####### FRAGMENT CLASS #######
from numpy.linalg import inv


class fragment:
    #####################################################################

    # needs to be edited (indexing)

    def __init__(self, impindx, Nsites, Nele, gen=False):
        if not gen:
            self.impindx = impindx
            # array defining index of impurity orbitals in site basis
            self.Nimp = impindx.shape[0]
            # number of impurity orbitals in fragment
            self.Nsites = Nsites
            # total number of sites (or basis functions if restricted) in total system
            self.Nele = Nele
            # total number of electrons in total system
            self.Nbasis = self.Nsites
            # number of basis functions in total system

            self.Ncore = int(Nele / 2) - self.Nimp
            # Number of core orbitals in fragment
            self.Nvirt = Nsites - 2 * self.Nimp - self.Ncore
            # Number of virtual orbitals in fragment

            self.gen = False

            # range of orbitals in embedding basis,
            # embedding basis always indexed as impurity, virtual, bath, core

            self.imprange = np.arange(0, self.Nimp)
            self.virtrange = np.arange(self.Nimp, self.Nimp + self.Nvirt)
            self.bathrange = np.arange(
                self.Nimp + self.Nvirt, 2 * self.Nimp + self.Nvirt
            )
            self.corerange = np.arange(2 * self.Nimp + self.Nvirt, self.Nsites)

            self.last_imp = self.Nimp
            self.last_virt = self.Nimp + self.Nvirt
            self.last_bath = 2 * self.Nimp + self.Nvirt
            self.last_core = self.Nsites

        if gen:
            self.impindx = impindx
            # array defining index of impurity orbitals in site basis
            self.Nimp = impindx.shape[0]
            # number of impurity orbitals in fragment
            self.Nsites = Nsites
            # total number of sites (or basis functions if restricted) in total system
            self.Nele = Nele
            # total number of electrons in total system
            self.Nbasis = 2 * self.Nsites
            # number of basis functions in total system

            self.Ncore = 2 * (int(Nele / 2) - int(self.Nimp / 2))
            # Number of core orbitals in fragment
            self.Nvirt = 2 * Nsites - 2 * self.Nimp - self.Ncore
            # Number of virtual orbitals in fragment

            self.gen = True

            # range of orbitals in embedding basis,
            # embedding basis always indexed as impurity, virtual, bath, core

            self.imprange = np.arange(0, self.Nimp)
            self.virtrange = np.arange(self.Nimp, self.Nimp + self.Nvirt)
            self.bathrange = np.arange(
                self.Nimp + self.Nvirt, 2 * self.Nimp + self.Nvirt
            )
            self.corerange = np.arange(2 * self.Nimp + self.Nvirt, 2 * self.Nsites)

            self.last_imp = self.Nimp
            self.last_virt = self.Nimp + self.Nvirt
            self.last_bath = 2 * self.Nimp + self.Nvirt
            self.last_core = self.Nsites * 2

    #####################################################################

    def get_rotmat(self, mf1RDM, gen=False):
        """
        Subroutine to generate rotation matrix from site to embedding basis
        PING currently impurities have to be listed in ascending order
        (though dont have to be sequential)
        """

        if not gen:
            # remove rows/columns corresponding to impurity sites from mf 1RDM
            mf1RDM = np.delete(mf1RDM, self.impindx, axis=0)
            mf1RDM = np.delete(mf1RDM, self.impindx, axis=1)

            # diagonalize environment part of 1RDM to obtain embedding
            # (virtual, bath, core) orbitals
            evals, evecs = np.linalg.eigh(mf1RDM)
            # print(f"non-zero part of R: \n {evecs}")
            # print(f"eigenvalues: \n {evals}")
            # form rotation matrix consisting of unit vectors
            # for impurity and the evecs for embedding
            # rotation matrix is ordered as impurity, virtual, bath, core

            # WORKS ONLY FOR MULTI-IMPURITY INDEXING'''

            """
            self.rotmat = np.zeros( [ self.Nsites, self.Nimp ] )
            for imp in range(self.Nimp):
                indx                     = self.impindx[imp]
                self.rotmat[ indx, imp ] = 1.0
                if indx <= evecs.shape[0]:
                    evecs = np.insert( evecs, indx, 0.0, axis=0 )
                else:
                    zero_coln = np.array([np.zeros(evecs.shape[1])])
                    evecs = np.concatenate((evecs, zero_coln), axis=0)

            self.rotmat = np.concatenate( (self.rotmat,evecs), axis=1 )
            """

            # WORKS FOR SINGLE IMPURITY INDEXING

            self.rotmat = np.zeros([self.Nsites, self.Nimp])

            for imp in range(self.Nimp):
                indx = self.impindx[imp]
                self.rotmat[indx, imp] = 1.0

            if self.impindx[0] > self.impindx[self.Nimp - 1]:
                for imp in range(self.Nimp):
                    rev_impindx = np.flipud(self.impindx)
                    indx = rev_impindx[imp]
                    if indx <= evecs.shape[0]:
                        evecs = np.insert(evecs, indx, 0.0, axis=0)
                    else:
                        print("index is out of range, attaching zeros in the end")
                        zero_coln = np.array([np.zeros(evecs.shape[1])])
                        evecs = np.concatenate((evecs, zero_coln), axis=0)
            else:
                for imp in range(self.Nimp):
                    indx = self.impindx[imp]
                    if indx <= evecs.shape[0]:
                        evecs = np.insert(evecs, indx, 0.0, axis=0)
                    else:
                        print("index is out of range, attaching zeros in the end")
                        zero_coln = np.array([np.zeros(evecs.shape[1])])
                        evecs = np.concatenate((evecs, zero_coln), axis=0)

            self.rotmat = np.concatenate((self.rotmat, evecs), axis=1)
            self.env1RDM_evals = evals

        if gen:
            # remove rows/columns corresponding to impurity sites from mf 1RDM
            mf1RDM = np.delete(mf1RDM, self.impindx, axis=0)
            mf1RDM = np.delete(mf1RDM, self.impindx, axis=1)

            # print(f"adjusted mf1RDM: \n {mf1RDM}")

            # diagonalize environment part of 1RDM to obtain embedding
            # (virtual, bath, core) orbitals
            evals, evecs = np.linalg.eigh(mf1RDM)

            # NOTE: blackberries; temporary check of reshaped R
            # permutation = [0, 3, 2, 1]
            # idx = np.empty_like(permutation)
            # idx[permutation] = np.arange(len(permutation))
            # block_evecs = utils.reshape_gtor_matrix(evecs)
            ## print(f"block evecs: \n {block_evecs}")
            ## print(
            ##   f"reshuffled indices, before spin staggering: \n {block_evecs[:,idx]}"
            ## )
            # evecs = utils.reshape_rtog_matrix(block_evecs[:, idx])
            # permutation = [0, 1, 3, 2]
            # idx = np.empty_like(permutation)
            # evals = evals[idx]
            # print(f"new evals: {evals}")
            # form rotation matrix consisting of unit vectors
            # for impurity and the evecs for embedding
            # rotation matrix is ordered as impurity, virtual, bath, core

            # WORKS ONLY FOR MULTI-IMPURITY INDEXING'''

            """
            self.rotmat = np.zeros( [ self.Nsites, self.Nimp ] )
            for imp in range(self.Nimp):
                indx                     = self.impindx[imp]
                self.rotmat[ indx, imp ] = 1.0
                if indx <= evecs.shape[0]:
                    evecs = np.insert( evecs, indx, 0.0, axis=0 )
                else:
                    zero_coln = np.array([np.zeros(evecs.shape[1])])
                    evecs = np.concatenate((evecs, zero_coln), axis=0)

            self.rotmat = np.concatenate( (self.rotmat,evecs), axis=1 )
            """

            # WORKS FOR SINGLE IMPURITY INDEXING

            self.rotmat = np.zeros([2 * self.Nsites, self.Nimp])

            for imp in range(self.Nimp):
                indx = self.impindx[imp]
                self.rotmat[indx, imp] = 1.0

            if self.impindx[0] > self.impindx[self.Nimp - 1]:
                for imp in range(self.Nimp):
                    rev_impindx = np.flipud(self.impindx)
                    indx = rev_impindx[imp]
                    if indx <= evecs.shape[0]:
                        evecs = np.insert(evecs, indx, 0.0, axis=0)
                    else:
                        print("index is out of range, attaching zeros in the end")
                        zero_coln = np.array([np.zeros(evecs.shape[1])])
                        evecs = np.concatenate((evecs, zero_coln), axis=0)
            else:
                for imp in range(self.Nimp):
                    indx = self.impindx[imp]
                    if indx <= evecs.shape[0]:
                        evecs = np.insert(evecs, indx, 0.0, axis=0)
                    else:
                        print("index is out of range, attaching zeros in the end")
                        zero_coln = np.array([np.zeros(evecs.shape[1])])
                        evecs = np.concatenate((evecs, zero_coln), axis=0)

            self.rotmat = np.concatenate((self.rotmat, evecs), axis=1)
            self.env1RDM_evals = evals

        # does this do anything bad? trying to use this function in rtog_transitions.py
        rotmat = self.rotmat
        env1RDM_evals = self.env1RDM_evals

        # np.set_printoptions(suppress=True)
        # print(f"new rotation matrix: \n {rotmat}")
        # blackbirds singing in the dead of night

        # if self.step == self.printstep:
        #    f = open("output_halffrag.txt", "a")
        #    f.write("\n new rotation matrix \n")
        #    f.close()
        #    utils.printarray(rotmat.real, "output_halffrag.txt")

        return rotmat, env1RDM_evals

    #####################################################################

    def get_Hemb(self, h_site, V_site, hamtype=0, hubsite_indx=None, gen=False):
        """
        Subroutine to the get the 1 and 2 e- terms of the
        Hamiltonian in the embedding basis
        Transformation accounts for interaction with the core
        Also calculates 1 e- term with only 1/2 interaction with the core -
        this is used in calculation of DMET energy
        """

        if not gen:
            # remove the virtual states from the rotation matrix
            # the rotation matrix is of form
            # ( site basis fcns ) x ( impurities, virtual, bath, core )
            rotmat_small = np.delete(
                self.rotmat, np.s_[self.Nimp : self.Nimp + self.Nvirt], 1
            )

            # rotate the 1 e- terms, h_emb currently
            # ( impurities, bath, core ) x ( impurities, bath, core )
            h_emb = utils.rot1el(h_site, rotmat_small)
            self.h_site = np.copy(h_site)

            # define 1 e- term of size ( impurities, bath ) x ( impurities, bath )
            # that will only have 1/2 interaction with the core
            self.h_emb_halfcore = np.copy(h_emb[: 2 * self.Nimp, : 2 * self.Nimp])

            # rotate the 2 e- terms
            if hamtype == 0:
                # General hamiltonian, V_emb currently
                # ( impurities, bath, core ) ^ 4
                V_emb = utils.rot2el_chem(V_site, rotmat_small)

            elif hamtype == 1:
                # Hubbard hamiltonian
                # remove core states from rotation matrix
                rotmat_vsmall = np.copy(rotmat_small[hubsite_indx, : 2 * self.Nimp])
                self.V_emb = V_site * np.einsum(
                    "ap,cp,pb,pd->abcd",
                    utils.adjoint(rotmat_vsmall),
                    utils.adjoint(rotmat_vsmall),
                    rotmat_vsmall,
                    rotmat_vsmall,
                )

            # augment the impurity/bath 1e- terms from contribution of coulomb
            # and exchange terms btwn impurity/bath and core
            # and augment the 1 e- term with only half the contribution
            # from the core to be used in DMET energy calculation

            if hamtype == 0:
                # General hamiltonian
                for core in range(2 * self.Nimp, 2 * self.Nimp + self.Ncore):
                    h_emb[: 2 * self.Nimp, : 2 * self.Nimp] = (
                        h_emb[: 2 * self.Nimp, : 2 * self.Nimp]
                        + 2 * V_emb[: 2 * self.Nimp, : 2 * self.Nimp, core, core]
                        - V_emb[: 2 * self.Nimp, core, core, : 2 * self.Nimp]
                    )

                    self.h_emb_halfcore += (
                        V_emb[: 2 * self.Nimp, : 2 * self.Nimp, core, core]
                        - 0.5 * V_emb[: 2 * self.Nimp, core, core, : 2 * self.Nimp]
                    )

            elif hamtype == 1:
                # Hubbard hamiltonian
                core_int = V_site * np.einsum(
                    "ap,pb,p->ab",
                    utils.adjoint(rotmat_vsmall),
                    rotmat_vsmall,
                    np.einsum(
                        "pe,ep->p",
                        rotmat_small[hubsite_indx, 2 * self.Nimp :],
                        utils.adjoint(rotmat_small[hubsite_indx, 2 * self.Nimp :]),
                    ),
                )
                h_emb[: 2 * self.Nimp, : 2 * self.Nimp] += core_int
                self.h_emb_halfcore += 0.5 * core_int

            # Calculate the energy associated with core-core interactions,
            # setting it numerically to a real number since it always will be
            Ecore = 0
            for core1 in range(2 * self.Nimp, 2 * self.Nimp + self.Ncore):
                Ecore += 2 * h_emb[core1, core1]

                if hamtype == 0:
                    # General hamiltonian
                    for core2 in range(2 * self.Nimp, 2 * self.Nimp + self.Ncore):
                        Ecore += (
                            2 * V_emb[core1, core1, core2, core2]
                            - V_emb[core1, core2, core2, core1]
                        )
            if hamtype == 1:
                # Hubbard hamiltonian
                vec = np.einsum(
                    "pe,ep->p",
                    rotmat_small[hubsite_indx, 2 * self.Nimp :],
                    utils.adjoint(rotmat_small[hubsite_indx, 2 * self.Nimp :]),
                )
                Ecore += V_site * np.einsum("p,p", vec, vec)

            self.Ecore = Ecore.real

            # Shrink h_emb and V_emb arrays to only include the impurity and bath
            self.h_emb = h_emb[: 2 * self.Nimp, : 2 * self.Nimp]
            if hamtype == 0:
                # General hamiltonian
                self.V_emb = V_emb[
                    : 2 * self.Nimp, : 2 * self.Nimp, : 2 * self.Nimp, : 2 * self.Nimp
                ]

        if gen:
            # remove the virtual states from the rotation matrix
            # the rotation matrix is of form
            # ( spinor basis fcns ) x ( impurities, virtual, bath, core )
            # deleting right-most indices first to avoid indexing issues
            # rotmat_small = np.delete(self.rotmat,
            #    np.s_[int(self.Nimp / 2 + self.Nsites) : int(
            #            self.Nimp / 2 + self.Nvirt / 2 + self.Nsites)],1)
            # deleting left-most indices
            # rotmat_small = np.delete(rotmat_small,
            #    np.s_[int(self.Nimp / 2) : int(self.Nimp / 2) + int(self.Nvirt / 2)], 1)
            rotmat_small = np.delete(
                self.rotmat, np.s_[self.Nimp : self.Nimp + self.Nvirt], 1
            )

            # rotate the 1 e- terms, h_emb currently
            # ( impurities, bath, core ) x ( impurities, bath, core )
            h_emb = utils.rot1el(h_site, rotmat_small)
            self.h_site = np.copy(h_site)

            # define 1 e- term of size ( impurities, bath ) x ( impurities, bath )
            # that will only have 1/2 interaction with the core
            # self.h_emb_halfcore = la.block_diag(
            #    np.copy(h_emb[: self.Nimp, : self.Nimp]),
            #    np.copy(h_emb[self.Nsites : self.Nimp + self.Nsites,
            #            self.Nsites : self.Nimp + self.Nsites]))
            self.h_emb_halfcore = np.copy(h_emb[: 2 * self.Nimp, : 2 * self.Nimp])

            # rotate the 2 e- terms
            hamtype = 0
            if hamtype == 0 or hamtype == 1:
                # General hamiltonian, V_emb currently
                # ( impurities, bath, core ) ^ 4
                V_emb = utils.rot2el_chem(V_site, rotmat_small)
            # elif hamtype == 1:
            # Hubbard hamiltonian
            # remove core states from rotation matrix
            #    rotmat_vsmall = np.copy(rotmat_small[hubsite_indx, : 2 * self.Nimp])
            #    self.V_emb = V_site * np.einsum(
            #        "ap,cp,pb,pd->abcd",
            #        utils.adjoint(rotmat_vsmall),
            #        utils.adjoint(rotmat_vsmall),
            #        rotmat_vsmall,
            #        rotmat_vsmall,
            #    )

            # augment the impurity/bath 1e- terms from contribution of coulomb
            # and exchange terms btwn impurity/bath and core
            # and augment the 1 e- term with only half the contribution
            # from the core to be used in DMET energy calculation

            # NOTE: possible issue here?
            if hamtype == 0 or hamtype == 1:
                # General (and Hubbard, for now) hamiltonian
                for core in range(2 * self.Nimp, 2 * self.Nimp + self.Ncore):
                    h_emb[: 2 * self.Nimp, : 2 * self.Nimp] = (
                        h_emb[: 2 * self.Nimp, : 2 * self.Nimp]
                        + V_emb[: 2 * self.Nimp, : 2 * self.Nimp, core, core]
                        - V_emb[: 2 * self.Nimp, core, core, : 2 * self.Nimp]
                    )
                    # NOTE: added an 0.5 to the first V_emb to account for lack of 2 above
                    self.h_emb_halfcore += (
                        0.5 * V_emb[: 2 * self.Nimp, : 2 * self.Nimp, core, core]
                        - 0.5 * V_emb[: 2 * self.Nimp, core, core, : 2 * self.Nimp]
                    )

            # elif hamtype == 1:
            # Hubbard hamiltonian
            #    core_int = V_site * np.einsum(
            #        "ap,pb,p->ab",
            #        utils.adjoint(rotmat_vsmall),
            #        rotmat_vsmall,
            #        np.einsum(
            #            "pe,ep->p",
            #            rotmat_small[hubsite_indx, 2 * self.Nimp :],
            #            utils.adjoint(rotmat_small[hubsite_indx, 2 * self.Nimp :]),
            #        ),
            #    )

            #    h_emb[: 2 * self.Nimp, : 2 * self.Nimp] += core_int
            #    self.h_emb_halfcore += 0.5 * core_int

            # Calculate the energy associated with core-core interactions,
            # setting it numerically to a real number since it always will be
            Ecore = 0
            for core1 in range(2 * self.Nimp, 2 * self.Nimp + self.Ncore):
                Ecore += h_emb[core1, core1]
                if hamtype == 0:
                    # General hamiltonian
                    for core2 in range(2 * self.Nimp, 2 * self.Nimp + self.Ncore):
                        Ecore += 0.5 * (
                            V_emb[core1, core1, core2, core2]
                            - V_emb[core1, core2, core2, core1]
                        )

            # if hamtype == 1:
            # Hubbard hamiltonian
            #    vec = np.einsum(
            #        "pe,ep->p",
            #        rotmat_small[hubsite_indx, 2 * self.Nimp :],
            #        utils.adjoint(rotmat_small[hubsite_indx, 2 * self.Nimp :]),
            #    )
            #    Ecore += V_site * np.einsum("p,p", vec, vec)

            self.Ecore = Ecore.real

            # Shrink h_emb and V_emb arrays to only include the impurity and bath
            self.h_emb = h_emb[: 2 * self.Nimp, : 2 * self.Nimp]
            if hamtype == 0:
                # General hamiltonian
                self.V_emb = V_emb[
                    : 2 * self.Nimp, : 2 * self.Nimp, : 2 * self.Nimp, : 2 * self.Nimp
                ]

        # if self.step == self.printstep:
        #    f = open("output_halffrag.txt", "a")
        #    f.write("\n hemb after shrinking \n")
        #    f.close()
        #    utils.printarray(self.h_emb.real, "output_halffrag.txt")
        #    f = open("output_halffrag.txt", "a")
        #    f.write(
        #        f"Vemb after shrinking: \n {self.V_emb[0,0,0,0]} \n Vemb shape: {self.V_emb.shape}"
        #    )
        #    f.close()

    #####################################################################

    def add_mu_Hemb(self, mu):
        # Subroutine to add a chemical potential, mu,
        # to only the impurity sites of embedding Hamiltonian
        for i in range(self.Nimp):
            self.h_emb[i, i] += mu

    #####################################################################

    def get_corr12RDM(self):
        # Subroutine to get the FCI 1RDM and 2RDM
        if not self.gen:
            self.corr1RDM, self.corr2RDM = fci_mod.get_corr12RDM(
                self.CIcoeffs, 2 * self.Nimp, (self.Nimp, self.Nimp)
            )
            self.full_corr1RDM = np.zeros([self.Nsites, self.Nsites])
            self.full_corr1RDM = self.full_corr1RDM.astype(complex)
            for c in self.corerange:
                self.full_corr1RDM[c][c] = 2
            corr1RDM_virt = np.insert(
                self.corr1RDM,
                self.Nimp,
                np.zeros((self.Nvirt, self.corr1RDM.shape[0])),
                0,
            )
            corr1RDM_virt = np.insert(
                corr1RDM_virt,
                self.Nimp,
                np.zeros((self.Nvirt, corr1RDM_virt.shape[0])),
                1,
            )

            self.full_corr1RDM[
                0 : 0 + corr1RDM_virt.shape[0], 0 : 0 + corr1RDM_virt.shape[1]
            ] += corr1RDM_virt
        if self.gen:
            self.corr1RDM, self.corr2RDM = fci_mod.get_corr12RDM(
                self.CIcoeffs, 2 * self.Nimp, self.Nimp, gen=True
            )
            self.full_corr1RDM = np.zeros([2 * self.Nsites, 2 * self.Nsites])
            self.full_corr1RDM = self.full_corr1RDM.astype(complex)
            for c in self.corerange:
                self.full_corr1RDM[c][c] = 1
            corr1RDM_virt = np.insert(
                self.corr1RDM,
                self.Nimp,
                np.zeros((self.Nvirt, self.corr1RDM.shape[0])),
                0,
            )
            corr1RDM_virt = np.insert(
                corr1RDM_virt,
                self.Nimp,
                np.zeros((self.Nvirt, corr1RDM_virt.shape[0])),
                1,
            )
            self.full_corr1RDM[
                0 : 0 + corr1RDM_virt.shape[0], 0 : 0 + corr1RDM_virt.shape[1]
            ] += corr1RDM_virt

    #####################################################################

    def eigvec_MF_check(self, mf1RDM):
        mf1RDM = np.delete(mf1RDM, self.impindx, axis=0)
        mf1RDM = np.delete(mf1RDM, self.impindx, axis=1)
        rotmat = np.delete(self.rotmat, self.impindx, axis=0)
        diag = utils.rot1el(mf1RDM, rotmat)
        identity = np.zeros((diag[0], diag[1]))
        print(
            "MF diagonalized by rotmat:",
            np.allclose(diag, identity, rtol=0, atol=1e-10),
        )
        print("difference between diagonalized and identity:", diag - identity)

    #####################################################################

    def get_frag_E(self):
        """
        Subroutine to calculate contribution to DMET energy from fragment
        Need to calculate embedding hamiltonian and 1/2 rdms prior to
        calling this routine
        Using democratic partitioning using Eq. 28 from  Wouters JCTC 2016
        This equation uses 1 e- part that only includes
        half the interaction with the core
        Notation for 1RDM is rho_pq = < c_q^dag c_p >
        Notation for 2RDM is gamma_pqrs = < c_p^dag c_r^dag c_s c_q >
        Notation for 1 body terms h1[p,q] = <p|h|q>
        Notation for 2 body terms V[p,q,r,s] = (pq|rs)
        """
        # Calculate fragment energy using democratic partitioning
        if not self.gen:
            self.Efrag = 0.0
            for orb1 in range(self.Nimp):
                for orb2 in range(2 * self.Nimp):
                    self.Efrag += (
                        self.h_emb_halfcore[orb1, orb2] * self.corr1RDM[orb2, orb1]
                    )
                    for orb3 in range(2 * self.Nimp):
                        for orb4 in range(2 * self.Nimp):
                            self.Efrag += 0.5 * (
                                self.V_emb[orb1, orb2, orb3, orb4]
                                * self.corr2RDM[orb1, orb2, orb3, orb4]
                            )

        if self.gen:
            self.Efrag = 0.0

            for orb1 in range(self.Nimp):
                for orb2 in range(2 * self.Nimp):
                    self.Efrag += (
                        self.h_emb_halfcore[orb1, orb2] * self.corr1RDM[orb2, orb1]
                    )
                    for orb3 in range(2 * self.Nimp):
                        for orb4 in range(2 * self.Nimp):
                            self.Efrag += 0.5 * (
                                self.V_emb[orb1, orb2, orb3, orb4]
                                * self.corr2RDM[orb1, orb2, orb3, orb4]
                            )
            np.set_printoptions(precision=5, suppress=True)

    #####################################################################

    def get_iddt_corr1RDM(
        self, h_site, V_site, hamtype=0, hubsite_indx=None, gen=False
    ):
        """
        Calculate the Hamiltonian commutator portion of
        the time-dependence of correlated 1RDM for each fragment
        ie { dot{ correlated 1RDM } } using notation from notes
        indexing in the embedding basis goes as
        ( impurities, virtual, bath, core )

        NOTE: Should be able to make this routine more efficient
        and it probably double calculates stuff from emb hamiltonian routine

        rotate the 2 e- terms into embedding basis - use notation
        MO (for molecular orbital) to distinguish
        from normal embedding Hamiltonian above which
        focuses just on impurity/bath region
        """
        iddt_corr_time = time.time()

        if hamtype == 0:
            # General hamiltonian
            V_MO = utils.rot2el_chem(V_site, self.rotmat)

        elif hamtype == 1:
            # Hubbard hamiltonian
            rotmat_Hub = self.rotmat[hubsite_indx, :]

        if not gen:
            # Form inactive Fock matrix
            if hamtype == 0:
                # General hamiltonian
                IFmat = utils.rot1el(h_site, self.rotmat)
                IFmat += 2.0 * np.einsum(
                    "abcc->ab", V_MO[:, :, self.corerange[:, None], self.corerange]
                )
                IFmat -= np.einsum(
                    "accb->ab", V_MO[:, self.corerange[:, None], self.corerange, :]
                )
                # print(f"rotmat: {self.rotmat}")
                # print(f"h_site: \n {h_site}")
                # print(f"IFmat: \n {IFmat}")

            elif hamtype == 1:
                # Hubbard hamiltonian
                IFmat = utils.rot1el(h_site, self.rotmat)
                tmp = np.einsum(
                    "pc,cp->p",
                    rotmat_Hub[:, self.corerange],
                    utils.adjoint(rotmat_Hub[:, self.corerange]),
                )
                IFmat += V_site * np.einsum(
                    "ap,pb,p->ab", utils.adjoint(rotmat_Hub), rotmat_Hub, tmp
                )
                # print(f"h_site: {h_site}")
                # print(f"IFmat: \n {IFmat}")

            # Form active Fock matrix
            actrange = np.concatenate((self.imprange, self.bathrange))
            if hamtype == 0:
                # General hamiltonian
                tmp = V_MO[:, :, actrange[:, None], actrange] - 0.5 * np.einsum(
                    "acdb->abdc", V_MO[:, actrange[:, None], actrange, :]
                )
                AFmat = np.einsum("cd,abdc->ab", self.corr1RDM, tmp)
                # print(f"AFmat: \n {AFmat}")

            elif hamtype == 1:
                # Hubbard hamiltonian
                tmp = np.einsum(
                    "pc,cd,dp->p",
                    rotmat_Hub[:, actrange],
                    self.corr1RDM,
                    utils.adjoint(rotmat_Hub[:, actrange]),
                )
                AFmat = (
                    0.5
                    * V_site
                    * np.einsum(
                        "ap,pb,p->ab", utils.adjoint(rotmat_Hub), rotmat_Hub, tmp
                    )
                )
                # print(f"AFmat: \n {AFmat}")
                # print(f"AFmat: \n {AFmat_temp}")

            # Form generalized Fock matrix from inactive and active ones
            if hamtype == 0:
                # General hamiltonian
                genFmat = np.zeros([self.Nsites, self.Nsites], dtype=complex)
                genFmat[self.corerange, :] = np.transpose(
                    2 * (IFmat[:, self.corerange] + AFmat[:, self.corerange])
                )
                # print(f"corerange: \n {self.corerange}")
                # print(f"invovled part of IFmat: \n {IFmat[:, self.corerange]}")
                # print(f"generalized F: \n {np.real(genFmat)}")
                genFmat[actrange, :] = np.transpose(
                    np.dot(IFmat[:, actrange], self.corr1RDM)
                )
                # print(f"active range: {actrange}")
                # print(f"corr1RDM: \n {self.corr1RDM}")
                # print(f"generalized F: \n {np.real(genFmat)}")
                # print(f"corr1RDM: \n {self.corr1RDM}")
                genFmat[actrange, :] += np.einsum(
                    "acde,bcde->ba",
                    V_MO[:, actrange[:, None, None], actrange[:, None], actrange],
                    self.corr2RDM,
                )
                # print(f"generalized Fock: \n {np.real(genFmat)}")

            elif hamtype == 1:
                # Hubbard hamiltonian
                genFmat = np.zeros([self.Nsites, self.Nsites], dtype=complex)
                genFmat[self.corerange, :] = np.transpose(
                    2 * (IFmat[:, self.corerange] + AFmat[:, self.corerange])
                )
                genFmat[actrange, :] = np.transpose(
                    np.dot(IFmat[:, actrange], self.corr1RDM)
                )
                tmp = np.einsum(
                    "dp,pc,pe,bcde->pb",
                    utils.adjoint(rotmat_Hub[:, actrange]),
                    rotmat_Hub[:, actrange],
                    rotmat_Hub[:, actrange],
                    self.corr2RDM,
                )
                genFmat[actrange, :] += V_site * np.transpose(
                    np.dot(utils.adjoint(rotmat_Hub), tmp)
                )

        if gen:
            if hamtype == 0 or hamtype == 1:
                # General hamiltonian
                IFmat = utils.rot1el(h_site, self.rotmat)
                IFmat += np.einsum(
                    "ijkk->ij", V_MO[:, :, self.corerange[:, None], self.corerange]
                )
                IFmat -= np.einsum(
                    "ikkj->ij", V_MO[:, self.corerange[:, None], self.corerange, :]
                )

                # print(f"h_site: {h_site}")
                # print(f"inactive Fock: \n {np.real(IFmat)}")

            # Form active Fock matrix
            actrange = np.concatenate((self.imprange, self.bathrange))
            if hamtype == 0 or hamtype == 1:
                # General hamiltonian
                # NOTE: why is actrange[:, None] not switched to align with l?
                #   what's the point of making it a column vector?
                tmp = V_MO[:, :, actrange[:, None], actrange] - np.einsum(
                    "iklj->ijlk", V_MO[:, actrange[:, None], actrange, :]
                )
                AFmat = np.einsum("kl,ijlk->ij", self.corr1RDM, tmp)
                # print(f"active Fock: \n {np.real(AFmat)}")

            # Form generalized Fock matrix from inactive and active ones
            if hamtype == 0 or hamtype == 1:
                # General hamiltonian
                genFmat = np.zeros([2 * self.Nsites, 2 * self.Nsites], dtype=complex)
                # if j in core:
                genFmat[self.corerange, :] = np.transpose(
                    IFmat[:, self.corerange] + AFmat[:, self.corerange]
                )
                # print(f"corerange: \n {self.corerange}")
                # print(f"invovled part of IFmat: \n {IFmat[:, self.corerange]}")
                # print(f"generalized Fock: \n {np.real(genFmat)}")
                # if j in impurity/bath:
                genFmat[actrange, :] = np.transpose(
                    np.dot(IFmat[:, actrange], self.corr1RDM)
                )
                temp_V = V_MO - np.einsum("imlk->iklm", V_MO)
                # print(f"active range: {actrange}")
                # print(f"corr1RDM: \n {self.corr1RDM}")
                # print(f"generalized Fock: \n {np.real(genFmat)}")
                genFmat[actrange, :] += 0.5 * np.einsum(
                    "iklm,jklm->ji",
                    temp_V[:, actrange[:, None, None], actrange[:, None], actrange],
                    self.corr2RDM,
                )
                # print(f"generalized Fock: \n {np.real(genFmat)}")

        # Calculate i times H commutator portion of time-dependence of corr1RDM
        self.iddt_corr1RDM = np.transpose(genFmat) - np.conjugate(genFmat)

        # temp, delete after debugging:
        self.genFmat = genFmat

        # if self.step == self.printstep:
        #    f = open("output_halffrag.txt", "a")
        #    f.write("\n generalized Fock matrix \n")
        #    f.close()
        #    utils.printarray(genFmat.real, "output_halffrag.txt", long_fmt=True)
        #    f = open("output_halffrag.txt", "a")
        #    f.write("TD of correlated 1RDM: \n")
        #    f.close()
        #    utils.printarray(
        #        self.iddt_corr1RDM.real, "output_halffrag.txt", long_fmt=True
        #    )

    #####################################################################

    def get_Xmat(self, mf1RDM, ddt_mf1RDM):
        # Subroutine to calculate the X-matrix to propagate embedding orbitals

        if not self.gen:
            # Initialize X-matrix
            self.Xmat = np.zeros([self.Nsites, self.Nsites], dtype=complex)

            # Index of orbitals in the site-basis corresponding to the environment
            # potential issue if considering non-sequantial
            # imp indx (single impurity)
            envindx = np.setdiff1d(np.arange(self.Nsites), self.impindx)

        if self.gen:
            self.Xmat = np.zeros([2 * self.Nsites, 2 * self.Nsites], dtype=complex)

            # Index of orbitals in the site-basis corresponding to the environment
            # potential issue if considering non-sequantial
            # imp indx (single impurity)
            # NOTE: PING not sure if this works... CHECK
            envindx = np.setdiff1d(np.arange(2 * self.Nsites), self.impindx)

        # Eigenvalues of environment part of mf1RDM
        env1RDM_evals = np.diag(
            np.real(
                utils.rot1el(
                    mf1RDM[envindx[:, None], envindx], self.rotmat[envindx, self.Nimp :]
                )
            )
        )

        self.mfevals = np.copy(env1RDM_evals)  # mrar

        # Calculate X-matrix setting diagonal, core-core,
        # and virtual virtual terms to zero

        # Matrix of one over the difference in
        # embedding orbital eigenvalues
        # Set redundant terms to zero,
        # ie diagonal, core-core, and virtual-virtual
        if not self.gen:
            eval_dif = np.zeros([self.Nsites - self.Nimp, self.Nsites - self.Nimp])
        if self.gen:
            eval_dif = np.zeros(
                [(2 * self.Nsites) - self.Nimp, (2 * self.Nsites) - self.Nimp]
            )

        # NOTE: this section of the code was not well checked; may have to go back and redo this

        # core-bath and core-virt
        for b in self.corerange:
            for a in np.concatenate((self.bathrange, self.virtrange)):
                if (
                    a != b
                    and np.abs(
                        env1RDM_evals[a - self.Nimp] - env1RDM_evals[b - self.Nimp]
                    )
                    > 1e-9
                ):
                    eval_dif[b - self.Nimp, a - self.Nimp] = 1.0 / (
                        env1RDM_evals[a - self.Nimp] - env1RDM_evals[b - self.Nimp]
                    )
        # bath-virt
        for b in self.bathrange:
            for a in self.virtrange:
                if (
                    a != b
                    and np.abs(
                        env1RDM_evals[a - self.Nimp] - env1RDM_evals[b - self.Nimp]
                    )
                    > 1e-9
                ):
                    eval_dif[b - self.Nimp, a - self.Nimp] = 1.0 / (
                        env1RDM_evals[a - self.Nimp] - env1RDM_evals[b - self.Nimp]
                    )

        # anti-symmetrize prior to calculating bath-bath portion
        eval_dif = eval_dif - eval_dif.T
        # bath-bath
        for b in self.bathrange:
            for a in self.bathrange:
                if (
                    a != b
                    and np.abs(
                        env1RDM_evals[a - self.Nimp] - env1RDM_evals[b - self.Nimp]
                    )
                    > 1e-9
                ):
                    eval_dif[b - self.Nimp, a - self.Nimp] = 1.0 / (
                        env1RDM_evals[a - self.Nimp] - env1RDM_evals[b - self.Nimp]
                    )

        # Rotate time-derivative of mean-field 1RDM
        self.Xmat[self.Nimp :, self.Nimp :] = utils.rot1el(
            1j * ddt_mf1RDM[envindx[:, None], envindx],
            self.rotmat[envindx, self.Nimp :],
        )

        # Multiply difference in eigenalues
        # and rotated time-derivative matrix
        self.Xmat[self.Nimp :, self.Nimp :] = np.multiply(
            eval_dif, self.Xmat[self.Nimp :, self.Nimp :]
        )
        self.Xmat = np.triu(self.Xmat) + np.triu(self.Xmat, 1).conjugate().transpose()

        # if self.step == self.printstep:
        #    f = open("output_halffrag.txt", "a")
        #    f.write("\n X matrix \n")
        #    f.close()
        #    utils.printarray(self.Xmat.real, "output_halffrag.txt")

    #####################################################################
