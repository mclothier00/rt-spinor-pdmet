#! /usr/bin/env python

"""
CASCI solver.

Features:
    RHF, UHF orbital based CASCI.
    RMP2, UMP2 natural orbital based CASCI.
    spin-averaged orbital based CASCI.
    split localization (using Jacobi rotation or CIAH newton optimizer).
    CAS basis matching / reordering for efficient restart calculation of DMET.

Author:
    Zhi-Hao Cui
    Bo-Xiao Zheng
"""
# ZHC TODO check PySCF's CASCI convention.

import os
import subprocess as sub
from tempfile import mkdtemp
import itertools as it

import numpy as np
import scipy.linalg as la

from libdmet_solid.solver import block, scf
from libdmet_solid.solver.scf import _get_veff
from libdmet_solid.system import integral
from libdmet_solid.lo import edmiston
from libdmet_solid.basis_transform.make_basis import find_closest_mo
from libdmet_solid.utils.munkres import Munkres, make_cost_matrix
from libdmet_solid.utils.misc import mdot, max_abs, grep, eri_idx, take_eri
from libdmet_solid.utils import logger as log

try:
    from libdmet_solid.solver import settings
except ImportError:
    import sys
    sys.stderr.write('settings.py not found.  Please create %s\n'
                     % os.path.join(os.path.dirname(__file__), 'settings.py'))
    raise ImportError('settings.py not found')

def check_sanity_cas(norbs, nelec, ncas, nelecas):
    """
    Sanity check on number of orbitals and electrons in the CAS space.
    
    Args:
        norbs: number of total orbitals.
        nelec: number of total electrons.
        ncas: number of CAS orbitals.
        nelecas: number of CAS electrons.
        all corresponds to per spin channel.
    """
    log.eassert(norbs >= 0, "norbs %s should >= 0", norbs)
    log.eassert(nelec >= 0, "nelec %s should >= 0", nelec)

    log.eassert(ncas >= 0,     "ncas:  %s should >= 0", ncas)
    log.eassert(ncas <= norbs, "ncas:  %s should <= norbs: %s", ncas, norbs)
    
    log.eassert(nelecas >= 0,     "nelecas:  %s should >= 0", nelecas)
    log.eassert(nelecas <= nelec, "nelecas:  %s should <= nelec: %s", nelecas, nelec)
    log.eassert(nelecas <= ncas,  "nelecas:  %s should <= ncas: %s", nelecas, ncas)

def cas_from_rdm1(rdm1, ncas, nelecas, nelec):
    """
    Define core, cas, virt and casinfo from rdm1.
    NOTE: this is for one spin sector.
    ncas, nelecas, nelec are for one spin sector
     
    Args:
        rdm1: rdm1, not spin traced. (norbs, norbs).
        ncas: number of CAS orbitals.
        nelec: number of total electrons.
        nelecas: number of CAS electrons.
        all corresponds to per spin channel.
    
    Returns:
        core, cas, virt orbitals: shape (norbs, nX), X = core, cas, virt
        casinfo: a tuple, number of orbitals within the cas, (nocc, npart, nvirt)
    """
    # natural orbitals:
    # reorder natural orbital, so that the orbital are from core, cas, virt
    natocc, natorb = la.eigh(rdm1)
    natocc = natocc[::-1]
    natorb = natorb[:, ::-1]
    log.debug(1, "Natural orbital occupations:\n%s", natocc)
    norbs = natocc.shape[0]
    check_sanity_cas(norbs, nelec, ncas, nelecas)
    
    # partition orbitals to core, cas, virt
    ncore = nelec - nelecas
    nvirt = norbs - ncore - ncas
    log.info("ncore = %d nvirt = %d ncas = %d", ncore, nvirt, ncas)
    log.info("core orbital occupation cut-off = %20.12f", \
            natocc[ncore-1] if ncore > 0 else 1)
    log.info("virt orbital occupation cut-off = %20.12f", \
            natocc[-nvirt] if nvirt > 0 else 0)
    log.eassert(ncore >= 0, "ncore: %s should >= 0", ncore)
    log.eassert(nvirt >= 0, "nvirt: %s should >= 0", nvirt)
    
    core = natorb[:, :ncore]
    cas  = natorb[:, ncore:norbs-nvirt]
    virt = natorb[:, norbs-nvirt:]
    
    # further partition the cas to cas_core, cas_partial, cas_virt
    # casinfo
    casocc = natocc[ncore:norbs-nvirt]
    _nvirt = np.sum(casocc < 0.3)
    _ncore = np.sum(casocc > 0.7)
    _npart = np.sum((casocc >= 0.3) * (casocc <= 0.7))
    log.info("In CAS:\n"
            "Occupied (n > 0.7): %d\n""Virtual  (n < 0.3): %d\n"
            "Partial Occupied: %d\n", _ncore, _nvirt, _npart)
    return core, cas, virt, (_ncore, _npart, _nvirt)

cas_from_1pdm = cas_from_rdm1

def cas_from_energy(mo_coeff, mo_energy, ncas, nelecas, nelec):
    """
    Define core, cas, virt and casinfo from rdm1.
    NOTE: this is for one spin sector.
    ncas, nelecas, nelec are for one spin sector
    
    Args:
        mo_coeff: MO coefficients, (norbs, norbs)
        mo_energy: MO energy, (norbs,)
        ncas: number of CAS orbitals.
        nelec: number of total electrons.
        nelecas: number of CAS electrons.
        all corresponds to per spin channel.
    
    Returns:
        core, cas, virt orbitals: shape (norbs, nX), X = core, cas, virt
        casinfo: a tuple, number of orbitals within the cas, (nocc, npart, nvirt)
    """
    log.debug(1, "Orbital energies:\n%s", mo_energy)
    norbs = mo_energy.shape[0]
    check_sanity_cas(norbs, nelec, ncas, nelecas) 

    ncore = nelec - nelecas
    nvirt = norbs - ncore - ncas
    log.info("ncore = %d nvirt = %d ncas = %d", ncore, nvirt, ncas)
    log.info("core orbital energy cut-off = %20.12f", \
            mo_energy[ncore-1] if ncore > 0 else float("Inf"))
    log.info("virt orbital eneryg cut-off = %20.12f", \
            mo_energy[-nvirt] if nvirt > 0 else -float("Inf"))
    log.eassert(ncore >= 0, "ncore: %s should >= 0", ncore)
    log.eassert(nvirt >= 0, "nvirt: %s should >= 0", nvirt)
    
    core = mo_coeff[:, :ncore]
    cas  = mo_coeff[:, ncore:norbs-nvirt]
    virt = mo_coeff[:, norbs-nvirt:]
    
    casenergy = mo_energy[ncore:norbs-nvirt]
    if nelecas == 0: # no electrons
        mu = casenergy[0] - 1e-3
    elif nelecas < ncas:
        mu = 0.5 * (casenergy[nelecas-1] + casenergy[nelecas])
        log.debug(0, "HF gap = %20.12f", casenergy[nelecas] - casenergy[nelecas-1])
    else: # all cas orbitals occupied
        mu = casenergy[-1] + 1e-3

    _nvirt = np.sum(casenergy > mu+1e-4)
    _ncore = np.sum(casenergy < mu-1e-4)
    _npart = np.sum((casenergy >= mu-1e-4) * (casenergy <= mu+1e-4))
    log.info("In CAS:\n"
            "Occupied (e < mu): %d\n""Virtual  (e > mu): %d\n"
            "Partial Occupied: %d\n", _ncore, _nvirt, _npart)
    return core, cas, virt, (_ncore, _npart, _nvirt)

def get_orbs(casci, Ham, guess, nelec, return_rdm1=False, scf_tol=1e-12):
    """
    Get CAS orbitals.

    Args:
        casci: CASCI object, will use options (MP2natorb, spinAverage, 
               scfsolver).
        Ham: integral object.
        guess: dm0 used for SCF guess (if RHF, spin traced).
        nelec: number of total electrons (include all spin channels).
        return_rdm1: whether return HF (or MP2) rdm1.
        scf_tol: tolerence for SCF.

    Returns:
        core, cas, virt orbitals: shape (norbs, nX), X = core, cas, virt
        casinfo: a list of tuple, number of orbitals within the cas, 
                 [(nocc, npart, nvirt) per spin]
        (rdm1): rdm1 of HF or MP2.
    """
    # ZHC TODO FIXME treat the case where Sz != 0
    # spin should be determined here?
    spin = Ham.H1["cd"].shape[0]
    
    # nelec, spin, bogo, res
    casci.scfsolver.set_system(nelec, casci.Sz, False, spin == 1, \
            max_memory=casci.max_memory)
    casci.scfsolver.set_integral(Ham)
    E_HF, rdm1_HF = casci.scfsolver.HF(tol=scf_tol, MaxIter=100, InitGuess=guess)
    rdm1 = rdm1_HF

    if casci.MP2natorb:
        E_MP2, rdm1_MP2 = casci.scfsolver.MP2()
        rdm1 = rdm1_MP2
        log.result("MP2 energy = %20.12f", E_HF + E_MP2)

    if casci.spinAverage:
        assert casci.Sz == 0
        core, cas, virt, casinfo = cas_from_rdm1(0.5 * (rdm1[0] + rdm1[1]), \
                casci.ncas, casci.nelecas // 2, nelec // 2)
    else:
        core    = [None for s in range(spin)]
        cas     = [None for s in range(spin)]
        virt    = [None for s in range(spin)]
        casinfo = [None for s in range(spin)]
        if casci.MP2natorb:
            for s in range(spin):
                log.info("Spin %d", s)
                if s == 0: # alpha
                    core[s], cas[s], virt[s], casinfo[s] = cas_from_rdm1(rdm1[s], \
                            casci.ncas, (casci.nelecas + casci.Sz) // 2, \
                            (nelec + casci.Sz) // 2)
                else: # beta
                    core[s], cas[s], virt[s], casinfo[s] = cas_from_rdm1(rdm1[s], \
                            casci.ncas, (casci.nelecas - casci.Sz) // 2, \
                            (nelec - casci.Sz) // 2)
        else:
            # use hartree-fock orbitals, we need orbital energy or order in this case
            mo = casci.scfsolver.get_mo()
            mo_energy = casci.scfsolver.get_mo_energy()
            for s in range(spin):
                log.info("Spin %d", s)
                if s == 0: # alpha
                    core[s], cas[s], virt[s], casinfo[s] = cas_from_energy(mo[s], \
                            mo_energy[s], casci.ncas, \
                            (casci.nelecas + casci.Sz) // 2, (nelec + casci.Sz) // 2)
                else: # beta
                    core[s], cas[s], virt[s], casinfo[s] = cas_from_energy(mo[s], \
                            mo_energy[s], casci.ncas, \
                            (casci.nelecas - casci.Sz) // 2, (nelec - casci.Sz) // 2)

        core = np.asarray(core)
        cas  = np.asarray(cas)
        virt = np.asarray(virt)
    if return_rdm1:
        return core, cas, virt, casinfo, rdm1
    else:
        return core, cas, virt, casinfo

def buildCASHamiltonian(Ham, core, cas):
    """
    Build CAS Hamiltonian.

    Args:
        Ham: Hamiltonian object
        core: core orbitals
        cas: cas orbitals

    Returns:
        casHam: Hamiltonian in the CAS space.
    """
    spin = Ham.H1["cd"].shape[0]
    nao = Ham.H1["cd"].shape[-1]
    # check the H2 format first
    eri_format, spin_dim = integral.get_eri_format(Ham.H2["ccdd"], nao)

    if spin > 1: # unrestricted
        if core.ndim == 2:
            core = np.asarray((core, core))
            cas  = np.asarray((cas, cas))
        rdm1_core = np.asarray((np.dot(core[0], core[0].conj().T), \
                                np.dot(core[1], core[1].conj().T)))
        # core-fock
        veff = _get_veff(rdm1_core, Ham.H2["ccdd"])
        
        # zero-energy
        # note the half factor for HF energy
        H0 = Ham.H0 + np.einsum("spq, sqp ->", (Ham.H1["cd"] + veff * 0.5), \
                rdm1_core)
         
        H1 = {
            "cd": np.asarray((
                mdot(cas[0].conj().T, Ham.H1["cd"][0] + veff[0], cas[0]),
                mdot(cas[1].conj().T, Ham.H1["cd"][1] + veff[1], cas[1])))
        }
        
        H2 = {
            "ccdd": scf.incore_transform(Ham.H2["ccdd"], \
                    (cas, cas, cas, cas), compact=(eri_format != 's1'))
        }
    else: # restricted
        if core.ndim == 2:
            core = core[None]
            cas = cas[None]
        rdm1_core = np.dot(core[0], core[0].conj().T)[None] * 2.0 # spin traced
        
        # veff
        veff = _get_veff(rdm1_core, Ham.H2["ccdd"])
        
        # zero-energy
        H0 = Ham.H0 + np.einsum("pq, qp ->", Ham.H1["cd"][0] + veff[0] * 0.5, \
                rdm1_core[0])

        H1 = {
            "cd": mdot(cas[0].conj().T, Ham.H1["cd"][0] + veff[0], cas[0])[None]
        }
        
        H2 = {
            "ccdd": scf.incore_transform(Ham.H2["ccdd"][0], \
                    (cas, cas, cas, cas), compact=(eri_format != 's1'))
        }
    return integral.Integral(cas.shape[2], spin==1, False, H0, H1, H2)

def split_localize(orbs, info, Ham, basis=None, Ham_eo=None, method='jacobi', \
        guess=None, tol=1e-10):
    """
    Split localization of the CAS orbitals.
    
    Args:
        orbs: CAS orbitals, C_eo_mo.
        info: CAS info.
        Ham: Ham in the CAS space.
        basis: embedding basis, C_lo_eo (spin, ncells, nlo, neo).
        Ham_eo: Ham in the original EO basis. 
        method: jacobi: use Jacobi rotation as ER optimizer, ciah: use CIAH as
                ER optimizer.
        guess: C_mo_lmo, used for ER localization guess.

    Returns:
        HamLocal: localized Hamiltonian.
        C_eo_lmo: C_eo_lmo.
        C_mo_lmo: C_mo_lmo.
        C_mo_lmo_no_reorder: C_mo_lmo without matching alpha and beta.
    """
    # ZHC TODO the transformation of Hamiltonian should only do once,
    # so the modification of Jacobi localizer is needed.
    spin = Ham.H1["cd"].shape[0]
    norbs = Ham.H1["cd"].shape[1]
    eri_format, spin_dim = integral.get_eri_format(Ham.H2["ccdd"], norbs)
    orbs = np.asarray(orbs)
    if orbs.ndim == 2:
        orbs = np.asarray((orbs,) * spin)
    C_eo_lmo = np.zeros_like(orbs)         # wrt original embedding basis
    C_mo_lmo = np.zeros_like(Ham.H1["cd"]) # wrt active orbitals
    
    if guess is not None:
        log.eassert(method != 'jacobi', "ER localization guess is not " 
                "supported by jacobi method, try use ciah instead")

    for s in range(spin):
        occ, part, virt = info[s]
        if occ > 0:
            log.info("Localization: Spin %d, occupied", s)
            if guess is None:
                guess_occ = None
            else:
                guess_occ = guess[s, :occ, :occ].copy()
            if method == 'jacobi': 
                # ZHC FIXME now only s1 is
                # supported for Jacobi rotation optimization
                localizer = edmiston.Localizer(take_eri(Ham.H2["ccdd"][s], \
                        range(occ), range(occ), range(occ), range(occ), \
                        compact=False))
                localizer.optimize(thr=tol)
                u_occ = localizer.coefs.T
                localizer = None
            else: 
                u_occ = edmiston.ER_model(mo_coeff=orbs[s, :, :occ], \
                        eri=Ham_eo.H2["ccdd"][s], guess=guess_occ, \
                        conv_tol=tol)[1]
            C_eo_lmo[s, :, :occ] = np.dot(orbs[s, :, :occ], u_occ)
            C_mo_lmo[s, :occ, :occ] = u_occ
        if virt > 0:
            log.info("Localization: Spin %d, virtual", s)
            if guess is None:
                guess_virt = None
            else:
                guess_virt = guess[s, -virt:, -virt:].copy()
            if method == 'jacobi':
                localizer = edmiston.Localizer(take_eri(Ham.H2["ccdd"][s], \
                        range(norbs-virt, norbs), range(norbs-virt, norbs), \
                        range(norbs-virt, norbs), range(norbs-virt, norbs), \
                        compact=False))
                localizer.optimize(thr=tol)
                u_virt = localizer.coefs.T
                localizer = None
            else:
                u_virt = edmiston.ER_model(mo_coeff=orbs[s, :, -virt:], \
                        eri=Ham_eo.H2["ccdd"][s], guess=guess_virt, \
                        conv_tol=tol)[1]
            C_eo_lmo[s, :, -virt:] = np.dot(orbs[s, :, -virt:], u_virt)
            C_mo_lmo[s, -virt:, -virt:] = u_virt
        if part > 0:
            log.info("Localization: Spin %d, partially occupied:", s)
            if guess is None:
                guess_part = None
            else:
                guess_part = guess[s, occ:norbs-virt, occ:norbs-virt].copy()
            if method == 'jacobi':
                localizer = edmiston.Localizer(take_eri(Ham.H2["ccdd"][s], \
                        range(occ, norbs-virt), range(occ, norbs-virt), \
                        range(occ, norbs-virt), range(occ, norbs-virt), \
                        compact=False))
                localizer.optimize(thr=tol)
                u_part = localizer.coefs.T
                localizer = None
            else: 
                u_part = edmiston.ER_model(mo_coeff=orbs[s, :, occ:norbs-virt], \
                        eri=Ham_eo.H2["ccdd"][s], guess=guess_part, \
                        conv_tol=tol)[1]
            C_eo_lmo[s, :, occ:norbs-virt] = np.dot(orbs[s,:, occ:norbs-virt], u_part)
            C_mo_lmo[s, occ:norbs-virt, occ:norbs-virt] = u_part
    
    C_mo_lmo_no_reorder = C_mo_lmo.copy()
    if spin > 1:
        if basis is not None:
            # match alpha, beta basis
            log.info("Match alpha and beta basis:")
            # ZHC NOTE: here I think we should NOT use find_closest_mo 
            # to match alpha and beta basis, since they may mix the beta
            # orbitals so that the localization is broken.
            #u_mat, diff = match_cas_basis(basis[[1]], C_eo_lmo[[1]], \
            #                              basis[[0]], C_eo_lmo[[0]], info[1])
            #C_eo_lmo[1] = np.dot(C_eo_lmo[1], u_mat[0])
            #C_mo_lmo[1] = np.dot(C_mo_lmo[1], u_mat[0])
            localbasis = np.asarray([
                    np.tensordot(basis[0], C_eo_lmo[0], (2, 0)),
                    np.tensordot(basis[1], C_eo_lmo[1], (2, 0))
            ])
            ovlp = np.tensordot(np.abs(localbasis[0]), np.abs(localbasis[1]), ((0,1), (0,1)))
            ovlp_sq = ovlp ** 2
            cost_matrix = make_cost_matrix(ovlp_sq, lambda cost: 1.0 - cost)
            m = Munkres()
            indices = m.compute(cost_matrix)
            indices = sorted(indices, key=lambda idx: idx[0])
            vals = [ovlp_sq[idx] for idx in indices]
            log.debug(1, "Orbital pairs and their overlap:")
            for i in range(norbs):
                log.debug(1, "(%2d, %2d) -> %12.6f", indices[i][0], indices[i][1], vals[i])
            log.info("Match localized orbitals: max %5.2f min %5.2f ave %5.2f", \
                    np.max(vals), np.min(vals), np.average(vals))

            # update C_eo_lmo and C_mo_lmo
            orderb = [idx[1] for idx in indices]
            C_eo_lmo[1] = C_eo_lmo[1][:, orderb]
            C_mo_lmo[1] = C_mo_lmo[1][:, orderb]

            localbasis[1] = localbasis[1][:, :, orderb]
            # make spin up and down basis have the same sign, i.e.
            # inner product larger than 1
            for i in range(norbs):
                if np.sum(localbasis[0, :, :, i] * localbasis[1, :, :, i]) < 0:
                    C_eo_lmo[1, :, i] *= -1.0
                    C_mo_lmo[1, :, i] *= -1.0

        H1 = {
            "cd":np.asarray([
                    mdot(C_mo_lmo[0].T, Ham.H1["cd"][0], C_mo_lmo[0]),
                    mdot(C_mo_lmo[1].T, Ham.H1["cd"][1], C_mo_lmo[1])
        ])}
        H2 = {
            "ccdd": scf.incore_transform(Ham.H2["ccdd"], \
                    (C_mo_lmo, C_mo_lmo, C_mo_lmo, C_mo_lmo), compact=(eri_format != 's1'))
        }
    else: # restricted
        H1 = {
            "cd":np.asarray([
                    mdot(C_mo_lmo[0].T, Ham.H1["cd"][0], C_mo_lmo[0])
        ])}
        H2 = {
            "ccdd": scf.incore_transform(Ham.H2["ccdd"][0], \
                    (C_mo_lmo, C_mo_lmo, C_mo_lmo, C_mo_lmo), compact=(eri_format != 's1'))
        }

    HamLocal = integral.Integral(norbs, spin == 1, False, Ham.H0, H1, H2)
    return HamLocal, C_eo_lmo, C_mo_lmo, C_mo_lmo_no_reorder

def gaopt(Ham, tmp="./tmp", only_file=False):
    """
    Generate reorder array using genetic algorithm.

    Args:
        Ham: integral
        tmp: the directory to generate input files and run GAOpt.
        only_file: only generate files and return K, without running GAOpt.

    Returns:
        order array if only_file is False, else return K.
    """
    norbs = Ham.norb
    # build K matrix
    K = np.empty((norbs, norbs))
    Int2e = Ham.H2["ccdd"]
    spin = Int2e.shape[0]
    eri_format, spin_dim = integral.get_eri_format(Int2e, norbs)
    IDX = lambda i, j, k, l: eri_idx(i, j, k, l, norbs, eri_format) 
    if spin > 1:
        for i, j in it.product(range(norbs), repeat=2):
            K[i, j]  = abs(Int2e[0][IDX(i, j, i, j)]) * 0.5 + \
                       abs(Int2e[1][IDX(i, j, i, j)]) * 0.5 + \
                       abs(Int2e[2][IDX(i, j, i, j)])
            K[i, j] += (abs(Ham.H1["cd"][0, i, j]) + \
                        abs(Ham.H1["cd"][1, i, j])) * 1e-7
    else: # restricted case
        for i, j in it.product(range(norbs), repeat=2):
            K[i, j]  = abs(Int2e[0][IDX(i, j, i, j)])
            K[i, j] += abs(Ham.H1["cd"][0, i, j]) * 1e-7

    # write K matrix
    wd = mkdtemp(prefix="GAOpt", dir=tmp)
    log.debug(0, "gaopt temporary file: %s", wd)
    with open(os.path.join(wd, "Kmat"), "w") as f:
        f.write("%d\n" % norbs)
        for i in range(norbs):
            for j in range(norbs):
                f.write(" %24.16f" % K[i, j])
            f.write("\n")

    # write configure file
    with open(os.path.join(wd, "ga.conf"), "w") as f:
        f.write("maxcomm 32\n")
        f.write("maxgen 20000\n")
        f.write("maxcell %d\n" % (norbs * 2))
        f.write("cloning 0.90\n")
        f.write("mutation 0.10\n")
        f.write("elite 1\n")
        f.write("scale 1.0\n")
        f.write("method gauss\n")
    
    executable = settings.GAOPTEXE
    log.debug(0, "gaopt executable: %s", executable)
    
    if not only_file:
        log.debug(0, "call gaopt")
        with open(os.path.join(wd, "output"), "w") as f:
            if block.Block.env_slurm:
            #if True:
                log.info( " ".join([ "srun", executable, "-s", "-config", os.path.join(wd, "ga.conf"), \
                     "-integral", os.path.join(wd, "Kmat")]))
                sub.check_call(" ".join([ "srun", \
                        executable, "-s", "-config", os.path.join(wd, "ga.conf"), \
                        "-integral", os.path.join(wd, "Kmat")]), stdout=f, shell=True)
            else:
                nproc = max(block.Block.nproc * block.Block.nnode, block.StackBlock.nproc \
                        * block.StackBlock.nthread * block.StackBlock.nnode)
                # ZHC FIXME mpirun seems not correct here for gaopt
                #sub.check_call(["mpirun", "-np", "%d" % nproc, \
                #        executable, "-s", "-config", os.path.join(wd, "ga.conf"), \
                #        "-integral", os.path.join(wd, "Kmat")], stdout = f)
                sub.check_call([executable, "-s", "-config", os.path.join(wd, "ga.conf"), \
                        "-integral", os.path.join(wd, "Kmat")], stdout=f)
        
        result = grep("DMRG REORDER FORMAT", os.path.join(wd, "output"), A=1).split("\n")[1]
        log.debug(1, "gaopt result: %s", result)
        reorder = [int(i)-1 for i in result.split(',')]
        sub.check_call(["rm", "-rf", wd])
        return reorder
    else:
        return K

def momopt(old_basis, new_basis):
    """
    Use Hungarian algorithm to match the basis.
    Find a set of reorder indices that makes
    new_basis[..., order] as close to old_basis as possible.
    
    Args:
        old_basis: (spin, ncells, nlo, neo)
        new_basis: (spin, ncells, nlo, neo).
    
    Returns:
        order: the reorder array, used to reorder the new basis.
        val: the average value of matching quality.
    """
    assert old_basis.ndim == 4
    factor = 1.0 / old_basis.shape[0]
    ovlp = factor * np.tensordot(np.abs(old_basis), np.abs(new_basis), \
            ((0, 1, 2), (0, 1, 2)))
    ovlp_sq = ovlp ** 2
    cost_matrix = make_cost_matrix(ovlp_sq, lambda cost: 1.0 - cost)

    m = Munkres()
    indices = m.compute(cost_matrix)
    indices = sorted(indices, key=lambda idx: idx[0])
    vals = [ovlp_sq[idx] for idx in indices]
    log.info("MOM reorder quality: max %5.2f min %5.2f ave %5.2f", \
            np.max(vals), np.min(vals), np.average(vals))

    order = [idx[1] for idx in indices]
    return order, np.average(vals)

def reorder(order, Ham, orbs, rot=None):
    """
    Reorder the Ham and orbs.

    Order 4 1 3 2 means 4 to 1, 1 to 2, 3 to 3, 2 to 4
    reorder in place.
    e.g. for H2, it is equivalent to:
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, order, :, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, order, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, order, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, :, order]
    
    Args:
        order: order array
        Ham: integral
        orbs: C_eo_lmo
        rot:  C_mo_lmo

    Returns:
        Ham, orbs (and rot) after reordering.
    """
    orbs = orbs[:, :, order] # spin, nao, nmo
    Ham.H1["cd"] = Ham.H1["cd"][:, order, :]
    Ham.H1["cd"] = Ham.H1["cd"][:, :, order]
    for s in range(Ham.H2["ccdd"].shape[0]):
        Ham.H2["ccdd"][s] = take_eri(Ham.H2["ccdd"][s], order, order, order, \
                order, compact=True)
    
    if rot is not None:
        rot = rot[:, :, order]
        return Ham, orbs, rot
    else:
        return Ham, orbs

def match_cas_basis(C_lo_eo, C_eo_mo, C_lo_eo_old, C_eo_mo_old, casinfo):
    """
    Match basis, split matching the occ, part and virt parts.
        
    Args:
        C_lo_eo: embedding basis, (spin, ncells, nlo, neo)
        C_eo_mo: cas orbitals, (spin, neo, nmo)
        C_lo_eo_old: old embedding basis, (spin, ncells, nlo, neo)
        C_eo_mo_old: old cas orbitals, (spin, neo, nmo)
        casinfo: casinfo (ncore, npart, nvirt)

    Returns:
        C_eo_mo_new: (spin, neo, nmo)
        diff_after: matching difference between the new basis and 
                    the old basis.
    """
    log.info("match CAS basis: ")
    spin, ncells, nlo, neo = C_lo_eo.shape
    spin, neo, nmo = C_eo_mo.shape
    
    C_lo_mo_old_col = []
    C_lo_mo_col = []
    u_mat = []
    for s in range(spin):
        nocc, npart, nvirt = casinfo[s]
        C_lo_mo = np.dot(C_lo_eo[s].reshape(ncells * nlo, neo), C_eo_mo[s])
        C_lo_mo_old = np.dot(C_lo_eo_old[s].reshape(ncells * nlo, neo), \
                C_eo_mo_old[s])
        C_lo_mo_col.append(C_lo_mo.reshape(ncells, nlo, nmo))
        C_lo_mo_old_col.append(C_lo_mo_old.reshape(ncells, nlo, nmo))
        
        # split matching
        if nocc > 0:
            u_occ = find_closest_mo(C_lo_mo[:, :nocc], \
                    C_lo_mo_old[:, :nocc], ovlp=None, return_rotmat=True)[1]
        else:
            u_occ = np.zeros((0, 0))
        
        if npart > 0:
            u_part = find_closest_mo(C_lo_mo[:, nocc:nocc+npart], \
                    C_lo_mo_old[:, nocc:nocc+npart], ovlp=None, \
                    return_rotmat=True)[1]
        else:
            u_part = np.zeros((0, 0))
        
        if nvirt > 0:
            u_virt = find_closest_mo(C_lo_mo[:, -nvirt:], \
                    C_lo_mo_old[:, -nvirt:], ovlp=None, return_rotmat=True)[1]
        else:
            u_virt = np.zeros((0, 0))
        u_mat.append(la.block_diag(u_occ, u_part, u_virt))

    u_mat = np.array(u_mat)
    C_lo_mo_old = np.array(C_lo_mo_old_col)
    C_lo_mo = np.array(C_lo_mo_col)
    C_eo_mo_new = np.einsum('sec, scC -> seC', C_eo_mo, u_mat)
    C_lo_mo_new = np.einsum('sRle, sec -> sRlc', C_lo_eo, C_eo_mo_new)
    
    diff_before = max_abs(C_lo_mo - C_lo_mo_old)
    diff_after = max_abs(C_lo_mo_new - C_lo_mo_old)
    log.info("difference before matching: %s", diff_before)
    log.info("difference after matching : %s", diff_after)
    return u_mat, diff_after

class CASCI(object):
    """
    CASCI solver.

    Features:
        RHF, UHF orbital based CASCI.
        RMP2, UMP2 natural orbital based CASCI.
        spin-averaged orbital based CASCI.
        split localization (using Jacobi rotation or CIAH newton optimizer).
        CAS basis matching / reordering for efficient restart calculation of DMET.
    """
    def __init__(self, ncas, nelecas, Sz=0, MP2natorb=False, spinAverage=False, \
            splitloc=True, cisolver=None, mom_reorder=True, tmpDir="./tmp", \
            loc_method='jacobi', loc_conv_tol=1e-10, max_memory=120000):
        """
        Args:
            ncas: number of CAS orbitals (spatial).
            nelecas: number of CAS electrons (all spin channel summed).
            Sz: spin = nelec_a - nelec_b.
            MP2natorb: use MP2 natural orbitals.
            spinAverage: use spin-averaged orbitals for UHF / UMP2 CASCI.
            splitloc: use split localization for CAS orbitals.
            cisolver: CAS space solver. cisolver can be customized as long as 
                      correct interface is implemented (run, run_dmet_Ham, 
                      cleanup), additional arguments are passed by ci_args in 
                      run function.
            mom_reorder: use MOM method to reorder the orbitals.
            tmpDir: temp directory.
            loc_method: jacobi: use Jacobi rotation optimizer; ciah: use CIAH
                        as optimizer.
            loc_conv_tol: tolerance for localization convergence.
        """
        log.eassert(ncas * 2 >= nelecas, \
                "CAS size (%s) not compatible with number of electrons %s", \
                ncas, nelecas)
        self.ncas = ncas
        self.nelecas = nelecas # alpha and beta
        self.Sz = Sz
        
        self.MP2natorb = MP2natorb
        self.spinAverage = spinAverage
        self.splitloc = splitloc
        self.loc_method = loc_method
        self.loc_conv_tol = loc_conv_tol
        self.max_memory = max_memory

        log.eassert(cisolver is not None, "No default ci solver is available" \
                " with CASCI, you have to use specify one.")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF(newton_ah=True)

        # reorder scheme for restart block DMRG calculations
        if mom_reorder:
            if block.Block.reorder:
                log.warning("Using maximal overlap method (MOM) to reorder localized "\
                        "orbitals, turning off Block reorder option")
                block.Block.reorder = False
        self.mom_reorder = mom_reorder
        self.tmpDir = tmpDir
        
        # store orbitals for basis matching
        self.basis_old     = None  # old embedding basis
        self.C_eo_mo_old   = None  # old CAS MO.
        # store u_matrix for localization
        self.C_mo_lmo_old  = None  # old u as new u0
        # store C_lo_lmo for orbital reorder
        self.C_lo_lmo_old  = None  # overall basis
        
        # store core and cas orbitals for run_dmet_ham 
        self.core = None
        self.cas  = None
    
    get_orbs = get_orbs

    def run(self, Ham, ci_args={}, guess=None, nelec=None, basis=None, \
            mom_tol=0.7, **kwargs): 
        """
        Main kernel of CASCI.

        Args:
            Ham: hamiltonian, assume H2 is in aaaa, bbbb, aabb order.
            ci_args: additional args for cisolver, 
                     a list or dict for ci solver, or None.
            guess: dm0 used for scfsolver (if RHF spin traced).
            nelec: number of total electrons (all spin channel summed),
                   if None, will use half-filled, i.e. Ham.norb
            basis: C_lo_eo, (spin, ncells, nlo, neo), used for basis matching.
            mom_tol: tolerance for MOM reorder quality.

        Returns:
            rdm1: in EO basis, same basis as Ham.
            E: energy directly from cisolver.
        """
        spin = Ham.H1["cd"].shape[0]
        if nelec is None: 
            nelec = Ham.norb
        
        # construct cas orbitals and casHam
        core, cas, virt, casinfo, dm = self.get_orbs(Ham, guess, nelec, return_rdm1=True)
        rdm1_core = np.array([np.dot(core[s], core[s].conj().T) for s in range(spin)])
        
        # check the basis shape are the same.
        basis_shape_unchanged = (self.basis_old is not None) and \
                                (self.basis_old.shape == basis.shape)
        # match cas orbital first.
        # and update C_eo_mo_old and basis_old.
        if basis_shape_unchanged:
            u_mat, diff = match_cas_basis(basis, cas, self.basis_old, \
                    self.C_eo_mo_old, casinfo)
            cas = np.array([np.dot(cas[s], u_mat[s]) for s in range(spin)])
        else:
            if self.basis_old is not None:
                log.warn("CASCI solver: basis shape changes [from %s to %s],\n"
                         "this is usually due to the truncation of bath. \n"
                         "Make sure the nelec (%s) is set correctly!", \
                         str(self.basis_old.shape), str(basis.shape), nelec)
        self.basis_old   = basis.copy()
        self.C_eo_mo_old = cas.copy()
        casHam = buildCASHamiltonian(Ham, core, cas)

        # split localization
        # and save the C_mo_lmo_old for the next step guess.
        if self.splitloc:
            if self.loc_method == 'jacobi':
                loc_guess = None
            else:
                # check whether the basis shape changes
                if basis_shape_unchanged:
                    loc_guess = self.C_mo_lmo_old
                else:
                    loc_guess = None
            casHam, cas, C_mo_lmo, C_mo_lmo_no_reorder = \
                    split_localize(cas, casinfo, casHam, \
                    basis=basis, Ham_eo=Ham, method=self.loc_method,
                    guess=loc_guess, tol=self.loc_conv_tol)
            self.C_mo_lmo_old = C_mo_lmo_no_reorder.copy()
        
        # MOM reorder compared with old basis
        if self.mom_reorder:
            log.eassert(basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            if self.C_lo_lmo_old is None or (not basis_shape_unchanged):
                order = gaopt(casHam, tmp=self.tmpDir)
                if "restart" in ci_args:
                    ci_args["restart"] = False
            else:
                # define C_lo_lmo
                C_lo_lmo = np.array([np.tensordot(basis[s], cas[s], (2, 0)) for s in range(spin)])
                # C_lo_lmo and self.C_lo_lmo_old are both in
                # atomic representation now
                order, q = momopt(self.C_lo_lmo_old, C_lo_lmo)
                # if the quality of mom is too bad, we reorder the orbitals
                # using genetic algorithm
                # FIXME seems larger q is a better choice
                if q < mom_tol:
                    order = gaopt(casHam, tmp=self.tmpDir)
                    if "restart" in ci_args:
                        ci_args["restart"] = False

            log.info("Orbital order: %s", order)
            # reorder casHam and cas
            casHam, cas = reorder(order, casHam, cas)
            # store cas in atomic basis
            self.C_lo_lmo_old = np.array([np.tensordot(basis[s], cas[s], (2, 0)) for s in range(spin)])
        
        # save cas orbital and core for interacting bath
        self.core = core
        self.cas  = cas 
        
        # initial dm0 guess for possible addtional mf in the cas solver.
        # assume ovlp is indentity.
        dm0_cas = np.array([mdot(self.cas[s].conj().T, dm[s], self.cas[s]) \
                for s in range(spin)])
        if spin == 1:
            dm0_cas *= 2.0
        if "dm0" not in ci_args:
            ci_args["dm0"] = dm0_cas
        
        # run CI solver, and rotate back dm, 
        # cisolver.run should return dm per spin channel.
        rdm1_cas, E = self.cisolver.run(casHam, nelec=self.nelecas, **ci_args)
        rdm1 = np.array([mdot(cas[s], rdm1_cas[s], cas[s].conj().T) \
                for s in range(spin)]) + rdm1_core
        
        # remove "dm0" for possible reuse
        ci_args.pop("dm0", None)
        return rdm1, E
    
    kernel = run

    def run_dmet_ham(self, Ham, ci_args={}, **kwargs):
        """
        Solve dmet hamiltonian with fixed wavefunction, but scaled Hamiltonian.
        
        Args:
            Ham: DMET scaled Ham.
            ci_args: additional args for cisolver, 
                     a list or dict for ci solver, or None
        
        Returns:
            E: DMET energy.
        """
        # rebuild the Ham since scaled Ham is different.
        casHam = buildCASHamiltonian(Ham, self.core, self.cas)
        E = self.cisolver.run_dmet_ham(casHam, **ci_args)
        # clean up for better memory footprint
        self._finalize()
        return E
    
    def _finalize(self):
        """
        Clean up ERI and twopdm for both CASCI object and attached cisolver.
        """
        import gc
        
        # mf for defining CAS
        self.scfsolver.mf._eri = None
        self.scfsolver.mf      = None
        
        # mp for defining CAS
        if hasattr(self.scfsolver, "mp"): 
            self.scfsolver.mp = None

        # if cisolver itself has scfsolver, clean up 
        if hasattr(self.cisolver, "scfsolver"):
            self.cisolver.scfsolver.mf._eri = None
            self.cisolver.scfsolver.mf      = None
        
        # if cisolver kernel has _scf, clean up 
        if hasattr(self.cisolver, "cisolver"):
            if hasattr(self.cisolver.cisolver, "_scf"):
                self.cisolver.cisolver._scf._eri = None
                self.cisolver.cisolver._scf      = None
        
        # remove the twopdm and twopdm_mo
        self.cisolver.twopdm_mo = None
        self.cisolver.twopdm    = None
        
        # manually release memory
        gc.collect()

    def cleanup(self):
        self.cisolver.cleanup()

DmrgCI = CASCI

