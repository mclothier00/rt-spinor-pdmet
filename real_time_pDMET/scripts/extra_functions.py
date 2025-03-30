##### from dynamics/fragment_mod.py #####

# def solve_GS(self):
# Use the embedding hamiltonian to solve for the FCI ground-state
#    if not self.gen:
#        self.CIcoeffs = fci_mod.FCI_GS(
#            self.h_emb,
#            self.V_emb,
#            self.Ecore,
#            2 * self.Nimp,
#            (self.Nimp, self.Nimp),
#        )
#    if self.gen:
#        self.CIcoeffs = fci_mod.FCI_GS(
#            self.h_emb,
#            self.V_emb,
#            self.Ecore,
#            2 * self.Nimp,
#            self.Nimp,
#            gen=True,
#        )

# def get_corr1RDM(self):
# Subroutine to get the FCI 1RDM
#    if not self.gen:
#        self.corr1RDM = fci_mod.get_corr1RDM(
#            self.CIcoeffs, 2 * self.Nimp, (self.Nimp, self.Nimp)
#        )
#    if self.gen:
#        self.corr1RDM = fci_mod.get_corr1RDM(
#            self.CIcoeffs, 2 * self.Nimp, self.Nimp, gen=True
#        )

# def static_corr_calc(
#    self, mf1RDM, mu, h_site, V_site, hamtype=0, hubsite_indx=None
# ):
# Subroutine to perform all steps of the static correlated calculation
# 1) get rotation matrix to embedding basis
#    self.get_rotmat(mf1RDM)
# 2) use rotation matrix to compute embedding hamiltonian
#    self.get_Hemb(h_site, V_site, hamtype, hubsite_indx)
# 3) add chemical potential to
# only impurity sites of embedding hamiltonian
#    self.add_mu_Hemb(mu)
# 4) perform corrleated calculation using embedding hamiltonian
#    self.solve_GS()
# 5) calculate correlated 1RDM
#    self.get_corr1RDM()
