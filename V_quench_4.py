import numpy as np
import real_time_pDMET.rtpdmet.static.pDMET_glob as static_driver
import real_time_pDMET.rtpdmet.static.transition as transition_driver
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.dynamics_driver as dynamic_driver
import real_time_pDMET.scripts.utils as utils

# restricted or generalized
gen = True

# set up system and static pdmet parameters
# NL = 29
# NR = 30
# Ndots = 1
# Nimp = 3 # fragment size
NL = 0
NR = 1
Ndots = 1
Nimp = 1  # fragment size
Nsites = NL + NR + Ndots
Nele = Nsites
Nfrag = int(Nsites / Nimp)
impindx = []
for i in range(Nfrag):
    impindx.append(np.arange(i * Nimp, (i + 1) * Nimp))

# generalized form of 2-electron terms in the embedding Hamiltonian
# hamtype = 0

# simplified tight-binding form of 2-electron terms in the embedding Hamiltonian
hamtype = 1

mubool = True
muhistory = True
Maxitr = 100000
tol = 1e-7
mf1RDM = None
# mf1RDM = np.load()

# static hamiltonian parameters
Full_static = True  # form a full 2-electron term
U_t0 = 0.0
Vg = 0.0
Vbias = 0.0
laser_t0 = False
boundary = False  # non-periodic hamiltonian

# dynamics variables
delt = 0.001
Nstep = 10000
Nprint = 10
init_time = 0.0
dG = 1e-5
dX = 1e-9
nproc = 1
integrator = "rk4"

# dynamic hamiltonian parameters
Full_dyn = True
U_dyn = 0.0
Vg_dyn = 0.0
Vbias_dyn = 0.01
laser_dyn = False
A_nott = 0.3
t_nott = 5
omega = 4.0
t_d = 0.5
laser_sites = list(range(0))
laser = False

# form a Hamiltonian for a static calculation

# Can specify a range or any other array with impurity indicies
hubb_indx = np.arange(NL, NL + Nimp)
timp = 1.0  # hopping parameter between two impurities
t_implead = 1.0
tleads = 1.0

# h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
#    Ndots,
#    NL,
#    NR,
#    Vg,
#    U_t0,
#    timp,
#    t_implead,
#    Vbias,
#    tleads,
#    boundary,
#    Full_static,
# )

h_site_r = np.asarray([[0, -1], [-1, 0.2]])

V_site_r = np.zeros([2, 2, 2, 2])

# run static calculation
static = static_driver.static_pdmet(
    Nsites,
    Nele,
    Nfrag,
    impindx,
    h_site_r,
    V_site_r,
    U_t0,
    Maxitr,
    mf1RDM,
    tol,
    hamtype,
    mubool,
    muhistory,
    hubb_indx,
)

static.kernel()

# transfer variables from static code to dynamics

if gen:
    system = transition_driver.rtog_transition(
        static,
        Nsites,
        Nele,
        Nfrag,
        impindx,
        h_site_r,
        V_site_r,
        hamtype,
        hubb_indx,
        boundary,
    )
else:
    system = transition_driver.rtor_transition(
        static,
        Nsites,
        Nele,
        Nfrag,
        impindx,
        h_site_r,
        V_site_r,
        hamtype,
        hubb_indx,
        boundary,
    )


print("Finished transition, beginning dynamics.")

hubb_indx = np.arange(NL, NL + Nimp)  # need to change, probably
timp = 1.0
t_implead = 1.0
tleads = 1.0


hamtype = 0

# print(f"before transition: \n {V_site_r}")
# run dynamics
if gen:
    # h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace_spinor(
    #    Ndots,
    #    NL,
    #    NR,
    #    Vg_dyn,
    #    U_dyn,
    #    timp,
    #    t_implead,
    #    Vbias_dyn,
    #    tleads,
    #    boundary,
    #    Full_dyn,
    # )

    Vb = 0.01 / 2

    h_site = np.asarray(
        [[Vb, 0, -1, 0], [0, Vb, 0, -1], [-1, 0, 0.2, 0], [0, -1, 0, 0.2]]
    )

    V_site = np.zeros([4, 4, 4, 4])

    dynamics = dynamic_driver.dynamics_driver(
        h_site,
        V_site,
        hamtype,
        system,
        delt,
        dG,
        dX,
        U_dyn,
        A_nott,
        t_nott,
        omega,
        t_d,
        nproc,
        Nstep,
        Nprint,
        integrator,
        hubb_indx,
        laser_sites,
        init_time,
        laser_dyn,
        gen,
    )
else:
    h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
        Ndots,
        NL,
        NR,
        Vg,
        U_dyn,
        timp,
        t_implead,
        Vbias_dyn,
        tleads,
        boundary,
        Full_dyn,
    )

    dynamics = dynamic_driver.dynamics_driver(
        h_site_r,
        V_site_r,
        hamtype,
        system,
        delt,
        dG,
        dX,
        U_dyn,
        A_nott,
        t_nott,
        omega,
        t_d,
        nproc,
        Nstep,
        Nprint,
        integrator,
        hubb_indx,
        laser_sites,
        init_time,
        laser_dyn,
    )

dynamics.kernel()
