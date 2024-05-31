# test change
import numpy as np
import sys
import math
import real_time_pDMET.rtpdmet.static.pDMET_glob as static_driver
import real_time_pDMET.rtpdmet.static.transition as transition_driver
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.dynamics_driver as dynamic_driver

# set up system and static pdmet parameters
NL = 29
NR = 30
Ndots = 1
Nimp = 4 # frangment size
Nsites = NL+NR+Ndots
Nele = Nsites
Nfrag = int(Nsites/Nimp)
impindx = []
for i in range(Nfrag):
    impindx.append(np.arange(i*Nimp, (i+1)*Nimp))

# generalized form of 2-electron terms in the embedding Hamiltonian
# hamtype = 0

# simplified tight-binding form of 2-electron  terms in the embedding Hamiltonian
hamtype = 1

mubool = True
muhistory = True
Maxitr = 100000
tol = 1e-7
mf1RDM = None
# mf1RDM = np.load()

# static hamiltonian parameters
Full_t0 = True# form a full 2-electron term
U = 0
Vg = 0.0
Vbias = False
laser_t0 = False
boundary = False #non-periodic hamiltonian

# dynamics variables
delt = 0.001
Nstep = 10000
Nprint = 10
init_time = 0.0
dG = 1e-11
dX = 1e-9
nproc = 1
integrator = 'rk4'

# dynamic hamiltonian parameters
Full_dyn = False
U_dyn = 3.0
Vg_dyn = 0.0
Vbias_dyn = 0.0
laser_dyn = True
A_nott = 0.3
t_nott = 5
omega = 4.0
t_d = 0.5
laser_sites = list(range(19, 40))
update = False

laser_pulse = (
    A_nott*np.exp(-((0-t_nott)**2)/(2*t_d**2))*math.cos(omega*(0-t_nott)))

# form a Hamiltonian for a static calculation

if Ndots == 1:
    hubb_indx = np.array([29])
    t = 1.0
    h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace_laser(
        Nsites, U, laser_t0, laser_sites, hubb_indx, t, update, boundary, Full_t0)
else:
    hubb_indx = np.arange(NL, NL+Nimp)
    t = 1.0
    h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace_laser(
        Nsites, U, laser_t0, laser_sites, hubb_indx, t, update, boundary, Full_t0)

print(h_site)
exit()
h_site=np.real(h_site)
V_site=np.real(V_site)

# run static calculation
static = static_driver.static_pdmet(
    Nsites, Nele, Nfrag, impindx, h_site,
    V_site, U, Maxitr, mf1RDM, tol,
    hamtype, mubool, muhistory, hubb_indx)

static.kernel()

# transfer variables from static code to dynamics
system = transition_driver.transition(
    static, Nsites, Nele, Nfrag, impindx,
    h_site, V_site, hamtype, hubb_indx, boundary)

# form a Hamiltonian for dynamics
if Ndots == 1:
    hubb_indx = np.array([29])
    t = 1.0
    h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace_laser(
        Nsites, U, laser_dyn, laser_sites, hubb_indx, t, update, boundary, Full_dyn)
else:
    hubb_indx = np.arange(NL, NL+Nimp)
    t = 1.0
    h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace_laser(
        Nsites, U, laser_dyn, laser_sites, hubb_indx, t, update, boundary, Full_dyn)

# run dynamics
dynamics = dynamic_driver.dynamics_driver(
    h_site, V_site, hamtype, system, delt, dG, dX, U_dyn, A_nott, t_nott, omega,
    t_d, nproc, Nstep, Nprint, integrator, hubb_indx, laser_sites, init_time,
    laser_dyn)

dynamics.kernel()
