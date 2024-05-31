import numpy as np
import sys
import math
import numpy as np
import real_time_pDMET.rtpdmet.static.pDMET_glob as static_driver
import real_time_pDMET.rtpdmet.static.transition as transition_driver
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.dynamics_driver as dynamic_driver

# set up system and static pdmet parameters
Nx = 8
Ny = 8
Nx_imp = 4 #fragment size in x
Ny_imp = 1 # frangment size in y
Ndots = 1 
Nsites = Nx * Ny
Nimp = Nx_imp * Ny_imp
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
Full_static = True # form a full 2-electron term
U_t0 = 0
Vg = 0.0
Vbias = 0.0
laser_t0 = False
boundary = False #non-periodic hamiltonian

# dynamics variables
delt = 0.001
Nstep = 10000
Nprint = 10
init_time = 0.0
dG = 1e-5
dX = 1e-9
nproc = 1
integrator = 'rk4'

# dynamic hamiltonian parameters
Full_dyn = False
U_dyn = 3.0
Vg_dyn = 0.0
Vbias_dyn = 0.0
laser_dyn = False
A_nott = 0.3
t_nott = 5
omega = 4.0
t_d = 0.5
laser_sites = list(range(0))
laser=False

# form a Hamiltonian for a static calculation

hubb_indx = np.array([35]) # impurity index

h_site, V_site = make_hams.make_2D_ham(
    Nx, Ny, Nx_imp, Ny_imp, U_t0, hubb_indx, boundary, Full_static)

# run static calculation
static = static_driver.static_pdmet(
    Nsites, Nele, Nfrag, impindx, h_site,
    V_site, U_t0, Maxitr, mf1RDM, tol,
    hamtype, mubool, muhistory, hubb_indx)

static.kernel()

# transfer variables from static code to dynamics
system = transition_driver.transition(
    static, Nsites, Nele, Nfrag, impindx,
    h_site, V_site, hamtype, hubb_indx, boundary)

# form a Hamiltonian for dynamics

h_site, V_site = make_hams.make_2D_ham(
    Nx, Ny, Nx_imp, Ny_imp, U_dyn, hubb_indx, boundary, Full_dyn)

# run dynamics
dynamics = dynamic_driver.dynamics_driver(
    h_site, V_site, hamtype, system, delt, dG, dX, U_dyn, A_nott, t_nott, omega,
    t_d, nproc, Nstep, Nprint, integrator, hubb_indx, laser_sites, init_time,
    laser_dyn)

dynamics.kernel()
