import numpy as np
import sys
import os
import real_time_pDMET.rtpdmet.dynamics.hartreefock as hartreefock
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.tdhf.with_fock_matrix.tdhf_wfock as tdhf

NL = 2
NR = 2
Ndots = 2
Nsites = NL+NR+Ndots
Nele = Nsites

hubsite_indx = np.arange(NL, NL+Ndots)

timp = 1.0
timplead = 1.0
tleads = 1.0
Vg = 0.0
Full = False

delt = 0.001
Nstep = 5000
Nprint = 1

# Initial Static Calculation
U = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, Full)

mf1RDM = hartreefock.rhf_calc_hubbard(Nele, h_site)

# Dynamics Calculation
U = 4.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, Full)
tdhf = tdhf.tdhf(
    Nsites, Nele, h_site, mf1RDM, delt, Nstep, Nprint, V_site, hubsite_indx)
tdhf.kernel()
