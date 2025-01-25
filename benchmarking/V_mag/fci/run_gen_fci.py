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
NR = 3
Ndots = 1
Nsites = NL + NR + Ndots
Nele = Nsites

timp = 1.0
timplead = 1.0
tleads = 1.0
Vg = 0.0
Full = True

delt = 0.0001
# Nstep = 5000
# Nprint = 100
Nstep = 500
Nprint = 20
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
Vbias = 0.01
gen = True

# magentic field, from Li paper
mag = -0.000085

mag_x = mag * np.sin(np.pi / 4) * np.cos(np.pi / 4)
mag_y = mag * np.sin(np.pi / 4) * np.sin(np.pi / 4)
mag_z = mag * np.cos(np.pi / 4)

h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

h_site1 = static_bfield.static_bfield(h_site_r, [mag_x, mag_y, mag_z])

h_site1 = np.kron(np.eye(2), h_site_r)
V_site1 = utils.block_tensor(V_site_r)
h_site = utils.reshape_rtog_matrix(h_site1)
V_site = utils.reshape_rtog_tensor(V_site1)

# Generalized Dynamics Calculation
tdfci = tdfci.tdfci(
    (Nsites * 2), Nele, h_site, V_site, CIcoeffs, delt, Nstep, Nprint, 0.0, gen
)
tdfci.kernel()
