import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import real_time_pDMET.tdfci.tdfci as tdfci
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod
import real_time_pDMET.scripts.utils as utils
import real_time_pDMET.rtpdmet.static.transition as transition
import numpy as np
import pyscf
import static_bfield as static_bfield


# NOTE: generalized TDFCI with generalized static FCI calculation

NL = 2
NR = 1
Ndots = 1
Nsites = NL + NR + Ndots
Nele = Nsites

timp = 1.0
timplead = 1.0
tleads = 1.0
Vg = 0.0
Full = True

delt = 0.0001
Nstep = 1000
Nprint = 50
boundary = False

# Initital Restricted Static Calculation
U = 0.0
Vbias = 0.0
gen = False

h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

CIcoeffs_res = fci_mod.FCI_GS(h_site, V_site, 0.0, Nsites, Nele, gen)

CIcoeffs = transition.to_gen_coeff(
    Nsites, Nsites, int(Nsites * 2), int(Nsites / 2), int(Nsites / 2), CIcoeffs_res
)

# Initializing Dynamics Calculation
U = 0.0
Vbias = 0.0
gen = True

h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

h_site1 = np.kron(np.eye(2), h_site_r)
V_site1 = utils.block_tensor(V_site_r)
V_site1[2, 2, 4, 4] = V_site1[4, 4, 6, 6] = V_site1[3, 3, 5, 5] = V_site1[
    5, 5, 7, 7
] = 0.1005
V_site1[2, 2, 5, 5] = V_site1[4, 4, 7, 7] = V_site1[3, 3, 4, 4] = V_site1[
    5, 5, 6, 6
] = -0.1005
V_site1[2, 3, 5, 4] = V_site1[4, 5, 7, 6] = V_site1[3, 2, 4, 5] = V_site1[
    5, 4, 6, 7
] = -2 * 0.1
h_site = utils.reshape_rtog_matrix(h_site1)
V_site = utils.reshape_rtog_tensor(V_site1)

# Generalized Dynamics Calculation
tdfci = tdfci.tdfci(
    (Nsites * 2), Nele, h_site, V_site, CIcoeffs, delt, Nstep, Nprint, 0.0, gen
)
tdfci.kernel()
