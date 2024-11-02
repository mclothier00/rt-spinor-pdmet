import numpy as np
import real_time_pDMET.rtpdmet.static.pDMET_glob as static_driver
import real_time_pDMET.rtpdmet.static.transition as transition_driver
import real_time_pDMET.scripts.make_hams as make_hams
import real_time_pDMET.rtpdmet.dynamics.dynamics_driver as dynamic_driver

# set up system and static pdmet parameters
# NL = 29
# NR = 30
# Ndots = 1
# Nimp = 3 # fragment size
NL = 2
NR = 3
Ndots = 1
Nimp = 3  # fragment size
Nsites = NL + NR + Ndots
Nele = Nsites
Nfrag = int(Nsites / Nimp)
impindx = []  # NOTE: I think this has the same dimension regardless
# of formalism; same number of sites just different number of orbitals
# per site
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
U_t0 = 0
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
laser = False

# form a Hamiltonian for a static calculation

if Ndots == 1:
    # hubb_indx = np.array([29]) # impurity index
    hubb_indx = np.array([5])  # impurity index
    t_implead = 1.0  # hopping parameter between a lead site and an impurity
    tleads = 1.0  # hopping parameter between two lead sites

    h_site_r, V_site_r = make_hams.make_ham_single_imp_anderson_realspace(
        NL, NR, Vg, U_t0, t_implead, Vbias, tleads, Full_static
    )
else:
    # Can specify a range or any other array with impurity indicies
    hubb_indx = np.arange(NL, NL + Nimp)
    timp = 1.0  # hopping parameter between two impurities
    t_implead = 1.0
    tleads = 1.0

    h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
        Ndots,
        NL,
        NR,
        Vg,
        U_t0,
        timp,
        t_implead,
        Vbias,
        hubb_indx,
        tleads,
        boundary,
        Full_static,
    )

# run static calculation
# NOTE: why does the static driver work with a generalized hamiltonian?
# FURTHER NOTE from present me: not sure past me was right; check initializing system with restricted
# and generalized hamiltonians
# FURTHER FURTHER NOTE from June me: confused as to why I bothered initializing with a gneeralized Ham at
# all and what previous present me could have been confused about; compare to Dariia's static code?
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

# creating new generalized hamiltonian from restricted hamiltonian
# currently done before transfering variables to dynamics to ensure totsystem has hopefully correct dynamics
h_site = np.kron(np.eye(2, dtype=int), h_site_r)
V_site = np.kron(np.eye(2, dtype=int), V_site_r)

# transfer variables from static code to dynamics
# system = transition_driver.transition(
#    static, Nsites, Nele, Nfrag, impindx,
#    h_site, V_site, hamtype, hubb_indx, boundary)


system = transition_driver.rtog_transition(
    static, Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, hubb_indx, boundary
)

# iddt_glob1RDM = mfmod.calc_iddt_glob1RDM(system)
# print(iddt_glob1RDM)

# G = mfmod.calc_Gmat(dG, system, iddt_glob1RDM)
# print(G)


# form a Hamiltonian for dynamics
if Ndots == 1:
    hubb_indx = np.array([5])
    t_implead = 1.0
    tleads = 1.0

    h_site_r, V_site_r = make_hams.make_ham_single_imp_anderson_realspace(
        NL, NR, Vg, U_dyn, t_implead, Vbias, tleads, Full_dyn
    )
else:
    hubb_indx = np.arange(NL, NL + Nimp)
    timp = 1.0
    t_implead = 1.0
    tleads = 1.0

    h_site_r, V_site_r = make_hams.make_ham_multi_imp_anderson_realspace(
        Ndots,
        NL,
        NR,
        Vg,
        U_dyn,
        timp,
        t_implead,
        Vbias,
        hubb_indx,
        tleads,
        boundary,
        Full_dyn,
    )


# NOTE: why is this done twice??
h_site = np.kron(np.eye(2, dtype=int), h_site_r)
V_site = np.kron(np.eye(2, dtype=int), V_site_r)

print(h_site)
exit()

# run dynamics
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
)

dynamics.kernel()
