import numpy
from pyscf import gto, scf, ao2mo
import rt_ghf

# NOTE: check ground state matches in both cases; energy and density matrix

filename = "hubbard_simple"
timestep = 0.001
total_steps = 50000
steps = 100


def _hubbard_hamilts_no_pbc(L, U):
    h1e = numpy.zeros((L, L))
    g2e = numpy.zeros((L,) * 4)
    for i in range(L - 1):
        h1e[i, (i + 1)] = h1e[(i + 1), i] = -1
    sites = int(L / 2)
    h1e[sites, (sites - 1)] = h1e[(sites - 1), sites] = 0
    return h1e, g2e


def rdm1_ghf_to_rhf(rdm1, sites):
    rdm1_aa = rdm1[:sites, :sites]
    rdm1_bb = rdm1[sites:, sites:]
    return rdm1_aa + rdm1_bb


sites = 4
L = int(
    sites * 2
)  # now (2 * number of sites), because working in generalized framework
# usite begins at 0
usite = 2
usite_beta = usite + sites
U = 0

mol = gto.M()
mol.nelectron = sites
mol.nao = L
mol.spin = 0
mol.incore_anyway = True
mol.build()

# set up hamiltonian
h1e, eri = _hubbard_hamilts_no_pbc(L, U)
mf = scf.GHF(mol)
mf.get_hcore = lambda *args: h1e
mf._eri = ao2mo.restore(1, eri, L)
mf.get_ovlp = lambda *args: numpy.eye(L)
mf.scf()

den = mf.make_rdm1()

print(f"mf1rdm: {rdm1_ghf_to_rhf(den, sites)}")

U = 4
# usite_beta = [x+sites for x in usite]
eri[usite_beta, usite_beta, usite, usite] = eri[
    usite, usite, usite_beta, usite_beta
] = U
eri[usite_beta, usite_beta, usite_beta, usite_beta] = eri[
    usite, usite, usite, usite
] = U
mf._eri = ao2mo.restore(1, eri, L)

usite = [usite + 1]

var = rt_ghf.GHF(mf, timestep, steps, total_steps, filename, usite, sites)

var.dynamics()
print("Dynamics complete, plotting results.")

var.plot_site_den()
