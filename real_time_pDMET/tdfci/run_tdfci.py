import real_time_pDMET.tdfci.tdfci as tdfci
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod

NL = 2
NR = 2
Ndots = 2
Nsites = NL+NR+Ndots
Nele = Nsites

timp = 1.0
timplead = 1.0
tleads = 1.0
Vg = 0.0
Full = True

delt = 0.001
Nstep = 5000
Nprint = 100
boundary = False
# Initital Static Calculation
U = 1.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full)

CIcoeffs = fci_mod.FCI_GS(h_site, V_site, 0.0, Nsites, Nele)

# Dynamics Calculation
U = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full)

tdfci = tdfci.tdfci(Nsites, Nele, h_site, V_site,
                    CIcoeffs, delt, Nstep, Nprint)
tdfci.kernel()
