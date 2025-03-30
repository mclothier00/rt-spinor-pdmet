##### functions currently unused within code #####


##### from dynamics_driver.py #####

    def import_check(self, tol):
       old_MF = np.copy(self.tot_system.mf1RDM)
       old_global = np.copy(self.tot_system.glob1RDM)
       self.tot_system.get_frag_corr1RDM()
       self.tot_system.get_glob1RDM()
       self.tot_system.get_nat_orbs()
       self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele / 2))
       check = print(np.allclose(old_MF, self.tot_system.mf1RDM, tol))
       check_glob = print(np.allclose(old_global, self.tot_system.glob1RDM, tol))
       if check is False:
           print("MF calculation doesnt agree up to the ", tol)
           print(old_MF - self.tot_system.mf1RDM)
           quit()
       if check_glob is False:
           print("Global calculation doesnt agree up to the ", tol)
           print(old_global - self.tot_system.glob1RDM)
           quit()

####################################

##### from mf1rdm_timedep_mod.py #####

def get_ddt_mf_NOs(system, G_site):
    ddt_mf1RDM = -1j * (np.dot(G_site, system.mf1RDM) - np.dot(system.mf1RDM, G_site))
    ddt_NOevecs = -1j * np.dot(G_site, system.NOevecs)

    return ddt_mf1RDM, ddt_NOevecs

####################################

##### from system_mod.py #####

    # def get_glob1RDM(self):
    #    """
    #    Subroutine to obtain global 1RDM formed from all fragments
    #    Need to have updated rotation matrices and correlated 1RDMs
    #    """

    # form global 1RDM forcing hermiticity

    # unpack necessary stuff
    # NOTE we're assuming all fragments have same number of impurities
    #    Nimp = self.frag_list[0].Nimp
    #    if np.iscomplexobj(self.frag_list[0].rotmat) or np.iscomplexobj(
    #        self.frag_list[0].corr1RDM
    #    ):
    #        rotmat_unpck = np.zeros([self.Nsites, 2 * Nimp, self.Nsites], dtype=complex)
    #        corr1RDM_unpck = np.zeros([2 * Nimp, self.Nsites], dtype=complex)
    #    else:
    #        rotmat_unpck = np.zeros([self.Nsites, 2 * Nimp, self.Nsites])
    #        corr1RDM_unpck = np.zeros([2 * Nimp, self.Nsites])
    #    for q in range(self.Nsites):
    # fragment for site q
    #        frag = self.frag_list[self.site_to_frag_list[q]]

    # index within fragment corresponding to site q -
    # note that q is an impurity orbital
    #        qimp = self.site_to_impindx[q]

    # unpack just impurity/bath parts of rotation matrix
    #        actrange = np.concatenate((frag.imprange, frag.bathrange))
    #        rotmat_unpck[:, :, q] = np.copy(frag.rotmat[:, actrange])

    # unpack necessary portion of corr1RDM
    #        corr1RDM_unpck[:, q] = np.copy(frag.corr1RDM[:, qimp])

    # calculate intermediate matrix
    #    tmp = np.einsum("paq,aq->pq", rotmat_unpck, corr1RDM_unpck)

    # form global 1RDM
    #    self.glob1RDM = 0.5 * (tmp + tmp.conj().T)

    #####################################################################

    # def get_nat_orbs(self):
    # Subroutine to obtain natural orbitals of global 1RDM
    #    NOevals, NOevecs = np.linalg.eigh(self.glob1RDM)

    # Re-order such that eigenvalues are in descending order
    #    self.NOevals = np.flip(NOevals)
    #    self.NOevecs = np.flip(NOevecs, 1)

    #####################################################################

    # def get_new_mf1RDM(self, Nocc):
    # Subroutine to obtain a new idempotent (mean-field) 1RDM from the
    # First Nocc natural orbitals of the global 1RDM
    # ie natural orbitals with the highest occupation

    #    NOocc = self.NOevecs[:, :Nocc]
    #    self.mf1RDM = 2.0 * np.dot(NOocc, NOocc.T.conj())

    #####################################################################

    # def static_corr_calc_wrapper(self, frag):
    # Subroutine that simply calls the static_corr_calc
    # subroutine for the given fragment class
    # The wrapper is necessary to parallelize using Pool

    #    frag.static_corr_calc(
    #        self.mf1RDM,
    #        self.mu,
    #        self.h_site,
    #        self.V_site,
    #        self.hamtype,
    #        self.hubsite_indx,
    #    )
    #    return frag

    #####################################################################

    # def corr_emb_calc(self, nproc, frag_pool):
    # Subroutine to perform full correlated calculation on each fragment
    # including transformations to embedding basis
    #    if not self.periodic:
    # non-periodic: calculate each fragment separately in parallel
    #        if nproc == 1:
    #            for frag in self.frag_list:
    #                frag.static_corr_calc(
    #                    self.mf1RDM,
    #                    self.mu,
    #                    self.h_site,
    #                    self.V_site,
    #                    self.hamtype,
    #                    self.hubsite_indx,
    #                )
    # else:
    #    frag_pool = multproc.Pool(nproc)
    #    self.frag_list = frag_pool.map(
    #    self.static_corr_calc_wrapper, self.frag_list)
    #    frag_pool.close()
    #    frag_pool.join()
    #    else:
    #        print("ERROR: Do not currently have periodic version of the code")
    #        exit()

    #####################################################################

    # def get_frag_corr1RDM(self):
    # Subroutine to calculate correlated 1RDM for each fragment
    #    for frag in self.frag_list:
    #        frag.get_corr1RDM()

    #####################################################################
    
    # def eigvec_frag_MF_check(self):
    #    for frag in self.frag_list:
    #        frag.eigvec_MF_check(self.mf1RDM)

####################################
