import numpy as np
import real_time_pDMET.rtpdmet.static.codes as codes
import real_time_pDMET.rtpdmet.static.fci_mod as fci_mod


class fragment():

    def __init__(
            self, impindx, Nsites, Nele, hubb_indx=None, mubool=False,
            delta=0.02, thrnele=1e-5, step=0.05):
        self.impindx = impindx
        self.Nsites = Nsites
        self.Nele = Nele
        self.hubb_indx = hubb_indx
        self.mubool = mubool
        # step size for chemical potential
        self.delta = delta
        # threshhold for the (current_electron/ideal electron) - 1
        # convergence of chemical potential
        self.thrnele = thrnele
        self.step = step
        self.Nimp = impindx.shape[0]
        self.Ncore = int(Nele/2) - self.Nimp
        self.Nvirt = Nsites - 2 * self.Nimp - self.Ncore
        self.imprange = np.arange(0, self.Nimp)
        self.virtrange = np.arange(self.Nimp, self.Nimp+self.Nvirt)
        self.bathrange = np.arange(
            self.Nimp+self.Nvirt, 2*self.Nimp+self.Nvirt)
        self.corerange = np.arange(2*self.Nimp+self.Nvirt, self.Nsites)

        self.last_imp = self.Nimp
        self.last_virt = self.Nimp + self.Nvirt
        self.last_bath = 2*self.Nimp + self.Nvirt
        self.last_core = self.Nsites
#####################################################################

    def initialize_RHF(self,  h_site, V_site):
        self.RDM = fci_mod.RHF(
            h_site, V_site, 2*self.Nimp, (self.Nimp, self.Nimp))
        return self.RDM
#####################################################################

    def get_rotmat(self, mf1RDM):
        # delete raws that correspond to impurity sites
        mf1RDM = np.delete(mf1RDM, self.impindx, axis=0)
        # delete columns that correspond to impurity sites
        mf1RDM = np.delete(mf1RDM, self.impindx, axis=1)
        # diagonalize environment part of 1RDM to obtain
        # embedding (virtual, bath, core) orbitals
        evals, evecs = np.linalg.eigh(mf1RDM)

        # form rotation matrix consisting of unit vectors
        # for impurity and the evecs for embedding
        # rotation matrix is ordered as impurity, virtual, bath, core
        self.rotmat = np.zeros([self.Nsites, self.Nimp])
        for imp in range(self.Nimp):
            indx = self.impindx[imp]
            self.rotmat[indx, imp] = 1.0

        if self.impindx[0] > self.impindx[self.Nimp-1]:
            for imp in range(self.Nimp):
                # ERR added for 1 imp hamiltonian
                rev_impindx = np.flipud(self.impindx)
                indx = rev_impindx[imp]
                if indx <= evecs.shape[0]:
                    evecs = np.insert(evecs, indx, 0.0, axis=0)
                    # incerting delta function component
                else:
                    print("index is  ut of range, attaching zeros in the end")
                    zero_coln = np.array([np.zeros(evecs.shape[1])])
                    evecs = np.concatenate((evecs, zero_coln), axis=0)
        else:
            for imp in range(self.Nimp):
                indx = self.impindx[imp]
                if indx <= evecs.shape[0]:
                    evecs = np.insert(evecs, indx, 0.0, axis=0)
                    # incerting delta function component
                else:
                    print("index is out of range, attaching zeros in the end")
                    zero_coln = np.array([np.zeros(evecs.shape[1])])
                    evecs = np.concatenate((evecs, zero_coln), axis=0)

        self.rotmat = np.concatenate((self.rotmat, evecs), axis=1)
        self.env1RDM_evals = evals
#####################################################################

    def get_Hemb(self, h_site, V_site, U, hamtype=0, hubsite_indx=None):
        ''' Subroutine to get 1e and 2e terms of Hamiltonian in embedding basis
            Transformation accounts for interaction with the core
            need to remove virtual orbtals from the rotation matrix
            initial form:
            ( site basis fcns ) x ( impurities, virtual, bath, core )'''
        rotmat_small = np.delete(
            self.rotmat, np.s_[self.Nimp:self.Nimp+self.Nvirt], 1)
        h_emb = codes.rot1el(h_site, rotmat_small)
        self.h_site = np.copy(h_site)
        self.h_emb_halfcore = np.copy(h_emb[:2*self.Nimp, :2*self.Nimp])
        if(hamtype == 0):
            V_emb = codes.rot2el_chem(V_site, rotmat_small)
        elif(hamtype == 1):
            rotmat_vsmall = np.copy(rotmat_small[hubsite_indx, :2*self.Nimp])
            self.V_emb = U*np.einsum(
                'ap,cp,pb,pd->abcd', codes.adjoint(rotmat_vsmall),
                codes.adjoint(rotmat_vsmall), rotmat_vsmall, rotmat_vsmall)

        if(hamtype == 0):
            for core in range(2*self.Nimp, 2*self.Nimp+self.Ncore):
                h_emb[:2*self.Nimp, :2*self.Nimp] = (
                    np.copy(h_emb[:2*self.Nimp, :2*self.Nimp]) +
                    2*V_emb[:2*self.Nimp, :2*self.Nimp, core, core] -
                    V_emb[:2*self.Nimp, core, core, :2*self.Nimp])

                self.h_emb_halfcore += (
                    V_emb[:2*self.Nimp, :2*self.Nimp, core, core]
                    - 0.5*V_emb[:2*self.Nimp, core, core, :2*self.Nimp])

        elif(hamtype == 1):
            core_int = U * np.einsum(
                'ap,pb,p->ab', codes.adjoint(rotmat_vsmall), rotmat_vsmall,
                np.einsum(
                    'pe,ep->p', rotmat_small[hubsite_indx, 2*self.Nimp:],
                    codes.adjoint(rotmat_small[hubsite_indx, 2*self.Nimp:])))

            h_emb[:2*self.Nimp, :2*self.Nimp] += core_int
            self.h_emb_halfcore += 0.5*core_int

        Ecore = 0
        for core1 in range(2*self.Nimp, 2*self.Nimp+self.Ncore):
            Ecore += 2*h_emb[core1, core1]

            if(hamtype == 0):
                for core2 in range(2*self.Nimp, 2*self.Nimp+self.Ncore):
                    Ecore += (2*V_emb[core1, core1, core2, core2]
                              - V_emb[core1, core2, core2, core1])

        if(hamtype == 1):
            vec = np.einsum(
                'pe,ep->p', rotmat_small[hubsite_indx, 2*self.Nimp:],
                codes.adjoint(rotmat_small[hubsite_indx, 2*self.Nimp:]))

            Ecore += V_site * np.einsum('p, p', vec, vec)
        self.Ecore = Ecore.real

        self.h_emb = h_emb[:2*self.Nimp, :2*self.Nimp]
        if(hamtype == 0):
            self.V_emb = V_emb[
                :2*self.Nimp, :2*self.Nimp, :2*self.Nimp, :2*self.Nimp]
#####################################################################

    def add_mu_Hemb(self, mu):
        # add a chemical potential, mu,
        # to only the impurity sites of embedding Hamiltonian
        for i in range(self.Nimp):
            self.h_emb[i, i] -= mu
####################################################################

    def solve_GS(self, U):
        # use embedding hamiltonian to solve for the FCI ground-state
        self.CIcoeffs, self.E_FCI = fci_mod.FCI_GS(
            self.h_emb, self.V_emb, U, 2*self.Nimp, (self.Nimp, self.Nimp))
        # calculation only on active space,
        # therefore 2*Nimp orbitals and 2 Nimp electrons
#####################################################################

    def get_corr1RDM(self):
        self.corr1RDM = fci_mod.get_corr1RDM(
            self.CIcoeffs, 2*self.Nimp, (self.Nimp+self.Nimp))
#####################################################################

    def get_corr12RDM(self):
        self.corr1RDM, self.corr2RDM = fci_mod.get_corr12RDM(
            self.CIcoeffs, 2*self.Nimp, (self.Nimp+self.Nimp))
#####################################################################

    def nele_in_frag(self):
        self.currNele = 0.0
        for e in range(self.Nimp):
            self.currNele += self.corr1RDM[e, e]
        return self.currNele
#####################################################################

    def corr_calc(self, mf1RDM, h_site, V_site, U, mu, hamtype=0,
                  hubb_indx=None, mubool=False):
        if (mubool):
            # get rotational matrix in embeding basis
            self.get_rotmat(mf1RDM)
            # compute emb hamiltonian with the rotational matrix
            self.get_Hemb(h_site, V_site, U, hamtype, hubb_indx)
            # add chem potential only to the impurity sites
            self.add_mu_Hemb(mu)
            self.solve_GS(U)
            self.get_corr1RDM()
            self.nele_in_frag()
            return self.currNele
        else:
            self.get_rotmat(mf1RDM)
            self.get_Hemb(h_site, V_site, U, hamtype, hubb_indx)
            # solve ground state by performing correlated
            # calculation with embedding hamiltonian
            self.solve_GS(U)
            self.get_corr1RDM()   # get correlated 1 RDM
#####################################################################

    def corr_calc_for_Nele(self, mf1RDM, h_site, V_site, U, mu,
                           hamtype=0, hubb_indx=None, mubool=False):
        self.get_Hemb(h_site, V_site, U, hamtype, hubb_indx)
        self.add_mu_Hemb(mu)
        self.solve_GS(U)
        self.get_corr1RDM()
        self.nele_in_frag()
        return self.corr1RDM, self.E_FCI, self.currNele
#####################################################################

    def get_frag_E(self):
        # calculate contribution to DMET energy from fragment
        self.Efrag = 0.0
        for orb1 in range(self.Nimp):
            for orb2 in range(2*self.Nimp):
                self.Efrag += (
                    self.h_emb_halfcore[orb1, orb2]
                    * self.corr1RDM[orb2, orb1])
                for orb3 in range(2*self.Nimp):
                    for orb4 in range(2*self.Nimp):
                        self.Efrag += (0.5 * self.V_emb[orb1, orb2, orb3, orb4]
                                       * self.corr2RDM[orb1, orb2, orb3, orb4])
