import numpy as np
from pyscf import gto, scf, ao2mo
from rt_scf import RT_SCF
import dmet_make_hams as make_hams
import dmet_utils as utils

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
Nstep = 20
Nprint = 20
boundary = False

# Initital Restricted Static Calculation

U = 0.0
Vbias = 0.0
gen = False

h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

mol = gto.M()
mol.nelectron = Nele
mol.nao = Nsites
mol.spin = 0
mol.build()

mf = scf.RHF(mol)

mf.get_hcore = lambda *args: h_site
mf._eri = ao2mo.restore(8, V_site, Nsites)
mf.get_ovlp = lambda *args: np.eye(Nsites)

mf.kernel()

den = mf.make_rdm1()

# Generalized Dynamics

h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

h_site = np.kron(np.eye(2), h_site_r)
V_site = utils.block_tensor(V_site_r)
V_site[2, 2, 4, 4] = V_site[4, 4, 6, 6] = V_site[3, 3, 5, 5] = V_site[5, 5, 7, 7] = (
    0.1005
)
V_site[2, 2, 5, 5] = V_site[4, 4, 7, 7] = V_site[3, 3, 4, 4] = V_site[
    5, 5, 6, 6
] = -0.1005
V_site[2, 3, 5, 4] = V_site[4, 5, 7, 6] = V_site[3, 2, 4, 5] = V_site[5, 4, 6, 7] = (
    -2 * 0.1
)

genden1 = np.kron(np.eye(2), den)
genden = utils.reshape_rtog_matrix(genden1)

mf = scf.addons.convert_to_ghf(mf)

mf.get_hcore = lambda *args: h_site
mf._eri = ao2mo.restore(1, V_site, int(Nsites * 2))
mf.get_ovlp = lambda *args: np.eye(int(Nsites * 2))
mf.dm = genden

rt_mf = RT_SCF(mf, 0.0001, 1.0)
rt_mf.observables.update(site_den=True, site_mag=True)

rt_mf.kernel()
