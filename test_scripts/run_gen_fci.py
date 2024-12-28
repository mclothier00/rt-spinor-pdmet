import real_time_pDMET.tdfci.tdfci as tdfci
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod
import real_time_pDMET.scripts.utils as utils
import numpy as np
import pyscf


# NOTE: generalized TDFCI with generalized static FCI calculation

# NL = 2
# NR = 2
# Ndots = 2
NL = 1
NR = 2
Ndots = 1
Nsites = NL + NR + Ndots
Nele = Nsites

timp = 1.0
timplead = 1.0
tleads = 1.0
Vg = 0.0
Full = True

delt = 0.001
# Nstep = 5000
# Nprint = 100
Nstep = 100
Nprint = 10
boundary = False

gen = True

# Initital Restricted Static Calculation
U = 0.0
Vbias = 0.0
h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

h_site1 = np.kron(np.eye(2), h_site_r)
V_site1 = utils.block_tensor(V_site_r)
h_site = utils.reshape_rtog_matrix(h_site1)
V_site = utils.reshape_rtog_tensor(V_site1)

CIcoeffs = fci_mod.FCI_GS(h_site, V_site, 0.0, (Nsites * 2), Nele, gen)

# Initializing Dynamics Calculation
U = 0.0
Vbias = -5.0
h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

h_site1 = np.kron(np.eye(2), h_site_r)
V_site1 = utils.block_tensor(V_site_r)
h_site = utils.reshape_rtog_matrix(h_site1)
V_site = utils.reshape_rtog_tensor(V_site1)


# Generalized Dynamics Calculation
tdfci = tdfci.tdfci(
    (Nsites * 2), Nele, h_site, V_site, CIcoeffs, delt, Nstep, Nprint, 0.0, gen
)
tdfci.kernel()
