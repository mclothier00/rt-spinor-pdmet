import real_time_pDMET.tdfci.tdfci as tdfci
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod
import numpy as np

NL = 1
NR = 1
Ndots = 0
Nsites = NL + NR + Ndots
Nele = Nsites

delt = 0.001
# Nstep = 5000
# Nprint = 100
Nstep = 100
Nprint = 10
boundary = False


h_site = np.asarray([[0, -1.0], [-1.0, 0]])
V_site = np.zeros((2, 2, 2, 2))
Nele = 2
Nsites = 2

CIcoeffs = fci_mod.FCI_GS(h_site, V_site, 0.0, Nsites, Nele)

h_site = np.asarray([[0.25, -1.0], [-1.0, -0.25]])
V_site = np.zeros((2, 2, 2, 2))

tdfci = tdfci.tdfci(Nsites, Nele, h_site, V_site, CIcoeffs, delt, Nstep, Nprint)
tdfci.kernel()
