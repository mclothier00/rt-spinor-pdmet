#!/usr/bin/env python

"""
Transform the PBC density fitting ERI integrals to the embedding space.

Author:
    Zhi-Hao Cui
    Tianyu Zhu
"""

import numpy as np
import scipy.linalg as la
import h5py

from pyscf import lib, ao2mo
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import gamma_point, member, unique, \
                                      KPT_DIFF_TOL
from pyscf.df import addons
from pyscf.pbc import df
from pyscf.pbc.df.df import _load3c
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos

from libdmet_solid.basis_transform.make_basis import multiply_basis
from libdmet_solid.system.lattice import get_phase, get_phase_R2k, round_to_FBZ
from libdmet_solid.utils.misc import mdot, max_abs, add_spin_dim
from libdmet_solid.utils import logger as log

ERI_IMAG_TOL = 1e-6

# *****************************************************************************
# External API
# *****************************************************************************

def get_emb_eri(cell, mydf, C_ao_lo=None, basis=None, unit_eri=False, \
        symmetry=1, t_reversal_symm=True, max_memory=None, swap_idx=None, \
        feri=None, kscaled_center=None, kconserv_tol=KPT_DIFF_TOL, **kwargs):
    """
    Get embedding ERIs with density fitting.
    
    Args:
        cell: cell object.
        mydf: density fitting object, can be: gdf, fft, aft, mdf.
        C_ao_lo: (spin, nkpts, nao, nlo), AO to LO basis in k.
        basis: (spin, ncells, nlo, nemb), embedding basis in R.
        unit_eri: only compute the unit ERI.
        symmetry: embedding ERI permutation symmetry.
        t_reversal_symm: whether to use time reversal symmetry.
        max_memory: maximum memory.
        swap_idx: swap the index in the integral for spin symmetry, depricated.
        feri: eri file name, will be depricate.
        kscaled_center: kpoint center shift.
        kconserv_tol: tolerance for k point conservation.
    
    Returns:
        eri: embedding ERI.
    """
    if isinstance(mydf, df.MDF):
        driver = get_emb_eri_fast_mdf
    elif isinstance(mydf, df.GDF):
        driver = get_emb_eri_fast_gdf
    elif isinstance(mydf, df.FFTDF):
        driver = get_emb_eri_fast_fft
    elif isinstance(mydf, df.AFTDF):
        driver = get_emb_eri_fast_aft
    else:
        raise ValueError("Unknown DF type for embedding ERI construction.")
    return driver(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, \
            feri=feri, kscaled_center=kscaled_center, symmetry=symmetry, \
            max_memory=max_memory, kconserv_tol=kconserv_tol, \
            unit_eri=unit_eri, swap_idx=swap_idx, \
            t_reversal_symm=t_reversal_symm)

def get_unit_eri(cell, mydf, C_ao_lo=None, \
        symmetry=1, t_reversal_symm=True, max_memory=None, swap_idx=None, \
        feri=None, kscaled_center=None, kconserv_tol=KPT_DIFF_TOL, **kwargs):
    """
    Get unit ERIs.
    
    See get_emb_eri for details.
    """
    C_ao_lo = np.asarray(C_ao_lo)
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    basis = np.empty_like(C_ao_lo)
    return get_emb_eri(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, feri=feri, \
        kscaled_center=kscaled_center, symmetry=symmetry, \
        max_memory=max_memory, kconserv_tol=kconserv_tol, unit_eri=True, \
        swap_idx=swap_idx, t_reversal_symm=t_reversal_symm, **kwargs)

# *****************************************************************************
# Functions for Fourier transformation, k-space to R-space, pack_tril
# *****************************************************************************

def get_basis_k(basis, phase_R2k):
    """
    Get FT transformed embedding basis.
    """
    spin = basis.shape[0]
    basis_k = np.empty_like(basis, dtype=np.complex128)
    for s in range(spin):
        basis_k[s] = lib.einsum('Rim, Rk -> kim', basis[s], phase_R2k)
    return basis_k

def _pack_tril(Lij):
    """
    Pack tril with possible spin dimension.
    """
    if Lij.ndim == 3:
        Lij_pack = lib.pack_tril(Lij)
    else:
        spin, nL, _, nao = Lij.shape
        nao_pair = nao * (nao+1) // 2
        Lij_pack = np.empty((spin, nL, nao_pair), dtype=Lij.dtype)
        for s in range(spin):
            lib.pack_tril(Lij[s], out=Lij_pack[s])
    return Lij_pack

def get_weights_t_reversal(cell, kpts, tol=KPT_DIFF_TOL):
    nkpts = len(kpts)
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    weights = np.ones(nkpts, dtype=np.int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    weights[i] = 2
                    weights[j] = 0
                    break
    assert np.sum(weights) == nkpts
    return weights

def get_naoaux(gdf):
    """
    The maximum dimension of auxiliary basis for every k-point.
    """
    if gdf._cderi is None:
        gdf.build()
    naux = -1
    for k in range(len(gdf.kpts)):
        # gdf._cderi['j3c/k_id/seg_id']
        with addons.load(gdf._cderi, 'j3c/%s'%k) as feri:
            if isinstance(feri, h5py.Group):
                naux_k = feri['0'].shape[0]
            else:
                naux_k = feri.shape[0]

        cell = gdf.cell
        if (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum' and
            not isinstance(gdf._cderi, np.ndarray)):
            with h5py.File(gdf._cderi, 'r') as feri:
                if 'j3c-/%s'%k in feri:
                    dat = feri['j3c-/%s'%k]
                    if isinstance(dat, h5py.Group):
                        naux_k += dat['0'].shape[0]
                    else:
                        naux_k += dat.shape[0]
        naux = max(naux, naux_k)
    return naux

# *****************************************************************************
# GDF ERI construction
# *****************************************************************************

def get_emb_eri_fast_gdf(cell, mydf, C_ao_lo=None, basis=None, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, unit_eri=False, swap_idx=None, \
        t_reversal_symm=True):
    """
    Fast routine to compute embedding space ERI on the fly.

    Args:
        C_ao_lo: (spin, nkpts, nao, nlo), AO to LO basis in k
        basis: (spin, ncells, nlo, nemb), embedding basis in R
        symmetry: embedding ERI symmetry
        max_memory: maximum memory
        t_reversal_symm: whether to use time reversal symmetry
    
    Returns:
        eri: embedding ERI.
    """
    log.info("Get ERI from Lpq in AO")

    # gdf variables
    nao = cell.nao_nr()
    kpts = mydf.kpts
    nkpts = len(kpts)
    if mydf._cderi is None:
        if feri is not None:
            mydf._cderi = feri
    if mydf.auxcell is None:
        # treat the possible drop of aux-basis at some kpts.
        naux = get_naoaux(mydf)
    else:
        naux = mydf.auxcell.nao_nr()
    
    # If C_ao_lo and basis not given, this routine is k2gamma AO transformation
    if C_ao_lo is None:
        C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        C_ao_lo[:, range(nao), range(nao)] = 1.0 # identity matrix for each k
    
    # add spin dimension for restricted C_ao_lo
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    
    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    
    # basis related
    if basis is None:
        basis = np.eye(nkpts * nao).reshape(1, nkpts, nao, nkpts * nao)
    if basis.shape[0] < C_ao_lo.shape[0]:
        basis = add_spin_dim(basis, C_ao_lo.shape[0])
    if C_ao_lo.shape[0] < basis.shape[0]:
        C_ao_lo = add_spin_dim(C_ao_lo, basis.shape[0])
    spin, _, _, nemb = basis.shape
    nemb_pair = nemb * (nemb+1) // 2
    
    if unit_eri: # unit ERI for DMFT
        C_ao_emb = C_ao_lo / (nkpts**0.75)
    else:
        phase = get_phase_R2k(cell, kpts)
        C_ao_emb = multiply_basis(C_ao_lo, get_basis_k(basis, phase)) \
                / (nkpts**(0.75)) # AO to embedding basis
    
    # ERI construction
    if t_reversal_symm:
        weights = get_weights_t_reversal(cell, kpts)
        log.debug(2, "time reversal symm used, weights of kpts:\n%s", weights)
        eri = np.zeros((spin * (spin+1) // 2, nemb_pair, nemb_pair)) # 4-fold
    else:
        weights = np.ones((nkpts,), dtype=np.int)
        log.debug(2, "time reversal symm not used.")
        eri = np.zeros((spin * (spin+1) // 2, nemb_pair, nemb_pair), \
                dtype=np.complex128) # 4-fold, complex

    if max_memory is None:
        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0]) * 0.9)
    blksize = max_memory * 1e6 / 16 / (nao**2 * 2)
    blksize = max(16, min(int(blksize), mydf.blockdim))
    Lij_s4 = np.empty((spin, naux, nemb_pair), dtype=np.complex128)
    buf = np.empty((spin * blksize * nemb_pair,), dtype=np.complex128)

    for kL in range(nkpts):
        if weights[kL] > 0:
            Lij_s4[:] = 0.0
            for i, kpti in enumerate(kpts):
                for j, kptj in enumerate(kpts):
                    # Find (ki,kj) that satisfies momentum conservation with kL
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    if max_abs(np.round(kconserv) - kconserv) < kconserv_tol:
                        log.debug(2, "kL: %5s / %-5s, ki: %-5s, kj: %-5s", \
                                kL + 1, nkpts, i, j)
                        log.debug(2, "Read Lpq and ao2emb")
                        # Loop over L chunks
                        step0, step1 = 0, 0
                        for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], \
                                max_memory=max_memory, compact=False, \
                                blksize=blksize):
                            Lpq = np.empty(LpqR.shape, dtype=np.complex128)
                            Lpq.real, Lpq.imag = LpqR, LpqI
                            LpqR = LpqI = None
                            lchunk = Lpq.shape[0]
                            step0, step1 = step1, step1 + lchunk
                            if swap_idx is None:
                                Lpq_beta = None
                            else:
                                swap_mesh = np.ix_(range(lchunk), swap_idx, \
                                        swap_idx)
                                Lpq_beta = Lpq.reshape(lchunk, nao, nao)\
                                        [swap_mesh].reshape(lchunk, nao*nao)
                            
                            Lij_loc = transform_ao_to_emb(Lpq, C_ao_emb, i, j,\
                                    Lpq_beta=Lpq_beta)
                            lib.pack_tril(Lij_loc.reshape(-1, nemb, nemb), \
                                    out=buf)
                            Lij_s4[:, step0:step1] += buf[:(spin * lchunk * \
                                    nemb_pair)].reshape(spin, lchunk, nemb_pair)

                        Lpq = Lpq_beta = Lij_loc = None
                        if step1 != naux:
                            log.warn("step1 (%s) != naux (%s), "
                            "auxbasis drop may happened...", step1, naux)
            log.debug(1, "ERI contraction for kL: %5s / %-5s ", kL + 1, nkpts)
            _Lij_s4_to_eri(Lij_s4, eri, weight=weights[kL], \
                    t_reversal_symm=t_reversal_symm)
    Lij_s4 = buf = None
    
    if not t_reversal_symm:
        eri_imag_norm = max_abs(eri.imag)
        log.info('ERI imaginary = %s', eri_imag_norm)
        if eri_imag_norm > ERI_IMAG_TOL:
            log.warn("ERI has imaginary part > %s (%s)", ERI_IMAG_TOL, \
                    eri_imag_norm)
        eri = eri.real

    log.debug(1, "ERI restore")
    eri = eri_restore(eri, symmetry, nemb)
    return eri

get_emb_eri_fast = get_emb_eri_fast_gdf

def transform_ao_to_emb(Lpq, basis, kp, kq, Lpq_beta=None):
    """
    Give Lpq (in AO basis) of shape (nL, nao*nao), 
    embedding basis after Fourier transform, kp, kq, 
    return Lij in embedding basis.

    Args:
        Lpq: AO integral.
        basis: C_ao_emb.
        kp: id of kp.
        kq: id of kq.
        Lpq_beta: beta integral.

    Returns:
        Lij: EO integral, shape (spin, nL, nemb*nemb).
    """
    if basis.ndim == 3:
        basis = basis[np.newaxis]
    spin, ncells, nlo, nemb = basis.shape
    if Lpq_beta is None:
        Lpq = [Lpq for s in range(spin)]
    else:
        Lpq = [Lpq, Lpq_beta]
    nL = Lpq[0].shape[0]
    Lij = np.empty((spin, nL, nemb*nemb), dtype=np.complex128)
    
    tao = []
    ao_loc = None
    for s in range(spin):
        mopq, pqslice = _conc_mos(basis[s, kp], basis[s, kq])[2:]
        _ao2mo.r_e2(Lpq[s], mopq, pqslice, tao, ao_loc, out=Lij[s])
    return Lij

def _Lij_s4_to_eri(Lij_s4, eri, weight=1, t_reversal_symm=False):
    """
    Contract 3-index tensor (L|ij) to 4-index tensor (ij|kl) 
    and accumulate it to eri.

    Args:
        Lij_s4: 4-fold symmetrized Lij, shape (nL, neo_pair).
        eri: resulting ERI.
        weight: kpt weight.
        t_reversal_symm: whether use time reversal symmetry.
    """
    if Lij_s4.ndim == 2:
        Lij_s4 = Lij_s4[np.newaxis]
    spin, nL, nemb_pair = Lij_s4.shape
    if t_reversal_symm:
        if spin == 1:
            Lij_loc = np.asarray(Lij_s4[0].real, order='C')
            if weight == 1:
                lib.dot(Lij_loc.T, Lij_loc, 1.0, eri[0], 1)
            elif weight == 2:
                lib.dot(Lij_loc.T, Lij_loc, 2.0, eri[0], 1)
                Lij_loc = np.asarray(Lij_s4[0].imag, order='C')
                lib.dot(Lij_loc.T, Lij_loc, 2.0, eri[0], 1)
            else:
                raise ValueError
        else:
            Lij_loc_a, Lij_loc_b = np.asarray(Lij_s4.real, order='C')
            if weight == 1:
                lib.dot(Lij_loc_a.T, Lij_loc_a, 1.0, eri[0], 1)
                lib.dot(Lij_loc_a.T, Lij_loc_b, 1.0, eri[1], 1)
                lib.dot(Lij_loc_b.T, Lij_loc_b, 1.0, eri[2], 1)
            elif weight == 2:
                lib.dot(Lij_loc_a.T, Lij_loc_a, 2.0, eri[0], 1)
                lib.dot(Lij_loc_a.T, Lij_loc_b, 2.0, eri[1], 1)
                lib.dot(Lij_loc_b.T, Lij_loc_b, 2.0, eri[2], 1)
                
                Lij_loc_a, Lij_loc_b = np.asarray(Lij_s4.imag, order='C')
                lib.dot(Lij_loc_a.T, Lij_loc_a, 2.0, eri[0], 1)
                lib.dot(Lij_loc_a.T, Lij_loc_b, 2.0, eri[1], 1) 
                lib.dot(Lij_loc_b.T, Lij_loc_b, 2.0, eri[2], 1)
            else:
                raise ValueError
    else:
        if spin == 1:
            lib.dot(Lij_s4[0].conj().T, Lij_s4[0], 1, eri[0], 1) # PL, LQ -> PQ
        else:
            lib.dot(Lij_s4[0].conj().T, Lij_s4[0], 1, eri[0], 1) # aaaa
            lib.dot(Lij_s4[0].conj().T, Lij_s4[1], 1, eri[1], 1) # aabb
            lib.dot(Lij_s4[1].conj().T, Lij_s4[1], 1, eri[2], 1) # bbbb

def eri_restore(eri, symmetry, nemb):
    """
    Restore eri with given permutation symmetry.
    """
    spin_pair = eri.shape[0]
    if spin_pair == 1:
        eri_res = ao2mo.restore(symmetry, eri[0].real, nemb)[np.newaxis]
    else:
        if symmetry == 4:
            nemb_pair = nemb*(nemb+1) // 2
            if eri.size == spin_pair * nemb_pair * nemb_pair:
                return eri.real.reshape(spin_pair, nemb_pair, nemb_pair)
            eri_res = np.empty((spin_pair, nemb_pair, nemb_pair))
        elif symmetry == 1:
            if eri.size == spin_pair * nemb**4:
                return eri.real.reshape(spin_pair, nemb, nemb, nemb, nemb)
            eri_res = np.empty((spin_pair, nemb, nemb, nemb, nemb))
        else:
            log.error("Spin unrestricted ERI does not support 8-fold symmetry.")
        for s in range(spin_pair):
            eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
    return eri_res

def get_unit_eri_fast_gdf(cell, mydf, C_ao_lo, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, swap_idx=None, t_reversal_symm=True):
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    basis = np.empty_like(C_ao_lo)
    return get_emb_eri_fast_gdf(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, feri=feri, \
        kscaled_center=kscaled_center, symmetry=symmetry, max_memory=max_memory, \
        kconserv_tol=kconserv_tol, unit_eri=True, swap_idx=swap_idx, \
        t_reversal_symm=t_reversal_symm)

get_unit_eri_fast = get_unit_eri_fast_gdf

# *****************************************************************************
# AFTDF ERI construction
# *****************************************************************************

def get_emb_eri_fast_aft(cell, mydf, C_ao_lo=None, basis=None, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, unit_eri=False, swap_idx=None, \
        t_reversal_symm=True):
    """
    Fast routine to compute embedding space ERI on the fly, by AFTDF
    C_ao_lo: (spin, nkpts, nao, nlo), transform matrix from AO to LO basis in k-space
    basis: (spin, ncells, nlo, nemb), embedding basis
    """
    log.info("Get ERI from AFTDF")

    # AFTDF variables
    nao = cell.nao_nr()
    kpts = mydf.kpts
    nkpts = len(kpts)
    
    # If C_ao_lo and basis not given, this routine is k2gamma AO transformation
    if C_ao_lo is None:
        C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        C_ao_lo[:, range(nao), range(nao)] = 1.0 # identity matrix for each k
    
    # add spin dimension for restricted C_ao_lo
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    
    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    
    # basis related
    if basis is None:
        basis = np.eye(nao*nkpts).reshape(1, nkpts, nao, nao*nkpts)
    if basis.shape[0] < C_ao_lo.shape[0]:
        basis = add_spin_dim(basis, C_ao_lo.shape[0])
    if C_ao_lo.shape[0] < basis.shape[0]:
        C_ao_lo = add_spin_dim(C_ao_lo, basis.shape[0])
    spin, _, _, nemb = basis.shape
    nemb_pair = nemb*(nemb+1) // 2
    
    if unit_eri:
        C_ao_emb = C_ao_lo / (nkpts**0.75)
    else:
        phase = get_phase_R2k(cell, kpts)
        C_ao_emb = multiply_basis(C_ao_lo, get_basis_k(basis, phase)) / (nkpts**(0.75)) # AO to embedding basis

    # ERI construction
    eri = np.zeros((spin*(spin+1)//2, nemb_pair, nemb_pair), dtype=np.complex128) # 4-fold symmetry
   
    # AFT variables
    kptij_lst = np.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:,0]
    kptjs_lst = kptij_lst[:,1]
    kpt_ji = kptjs_lst - kptis_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    ngrids = np.prod(mydf.mesh)
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    
    if max_memory is None:
        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0]) * 0.9)

    for uniq_id, kpt in enumerate(uniq_kpts):
        log.debug(1, "q: %s / %s", uniq_id+1, len(uniq_kpts))
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = np.where(uniq_inverse == uniq_id)[0]
        kptjs = kptjs_lst[adapted_ji_idx]
        coulG = mydf.weighted_coulG(q, False, mydf.mesh)
        #coulG *= factor

        # IJ part 
        log.debug(2, "Calculate zij")
        zij_R = np.zeros((spin, ngrids, nemb*nemb), dtype=np.complex128)
        moij_list = []
        ijslice_list = []
        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            moij, ijslice = zip(*[_conc_mos(C_ao_emb[s][ki], C_ao_emb[s][kj])[2:] for s in range(spin)])
            moij_list.append(moij)
            ijslice_list.append(ijslice)
        
        # ZHC FIXME TODO: coulG should be able to separate
        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, kptjs, max_memory=max_memory):
            for ji, aoao in enumerate(aoaoks):
                ki = adapted_ji_idx[ji] // nkpts
                kj = adapted_ji_idx[ji] %  nkpts
                
                buf = aoao.transpose(1,2,0).reshape(nao**2, p1-p0)
                zij = transform_ao_to_emb_aft(lib.transpose(buf), C_ao_emb, \
                        moij_list[ji], ijslice_list[ji])
                zij *= coulG[p0:p1,None]
                zij_R[:, p0:p1] += zij
        
        coulG = buf = aoao = zij = None
        zij_R = _pack_tril(zij_R.reshape(spin, -1, nemb, nemb))
        
        # KL part
        log.debug(2, "Calculate zkl")
        zkl_R = np.zeros((spin, ngrids, nemb*nemb), dtype=np.complex128)
        mokl_list = []
        klslice_list = []
        for kk in range(nkpts):
            kl = kconserv[ki, kj, kk]
            mokl, klslice = zip(*[_conc_mos(C_ao_emb[s][kk], C_ao_emb[s][kl])[2:] for s in range(spin)])
            mokl_list.append(mokl)
            klslice_list.append(klslice)

        ki = adapted_ji_idx[0] // nkpts
        kj = adapted_ji_idx[0] % nkpts
        kptls = kpts[kconserv[ki, kj, :]]
        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, -kptls, max_memory=max_memory):
            for kk, aoao in enumerate(aoaoks):
                buf = aoao.conj().transpose(1,2,0).reshape(nao**2,p1-p0)
                zkl = transform_ao_to_emb_aft(lib.transpose(buf), C_ao_emb, \
                        mokl_list[kk], klslice_list[kk])
                zkl_R[:, p0:p1] += zkl

        buf = aoao = zkl = None
        zkl_R = _pack_tril(zkl_R.reshape(spin, -1, nemb, nemb))
        
        log.debug(1, "Contract ERI")
        _Lij_s4_to_eri_aft(zij_R, zkl_R, eri)
    
    zij_R = zkl_R = None
        
    eri_imag_norm = max_abs(eri.imag)
    log.info('ERI imaginary = %s', eri_imag_norm)
    if eri_imag_norm > ERI_IMAG_TOL:
        log.warn("ERI has imaginary part > %s (%s)", ERI_IMAG_TOL, \
                eri_imag_norm)
    eri = eri.real

    log.debug(1, "ERI restore")
    eri = eri_restore(eri, symmetry, nemb)
    return eri

def get_unit_eri_fast_aft(cell, mydf, C_ao_lo, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, swap_idx=None, t_reversal_symm=True):
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    basis = np.empty_like(C_ao_lo)
    return get_emb_eri_fast_aft(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, feri=feri, \
        kscaled_center=kscaled_center, symmetry=symmetry, max_memory=max_memory, \
        kconserv_tol=kconserv_tol, unit_eri=True, swap_idx=swap_idx, \
        t_reversal_symm=t_reversal_symm)

def transform_ao_to_emb_aft(Lij, basis, moij, ijslice):
    """
    Give Lij (in AO basis) of shape (nL, nao*nao), embedding basis after FT, ki, kj, 
    return Lmn in embedding basis.
    Lmn is of shape (spin, nL, nemb*nemb)
    """

    if basis.ndim == 2:
        basis = basis[np.newaxis]
    spin, ncells, nlo, nemb = basis.shape
    nL = Lij.shape[-2]
    Lmn = np.empty((spin, nL, nemb*nemb), dtype=np.complex128)
    
    tao = []
    ao_loc = None
    for s in range(spin):
        _ao2mo.r_e2(Lij, moij[s], ijslice[s], tao, ao_loc, out=Lmn[s])
    return Lmn

def _Lij_s4_to_eri_aft(Lij_s4, Lkl_s4, eri):
    """
    Contract 3-index tensor (L|ij) to 4-index tensor (ij|kl) and accumulate it to eri.
    """
    if Lij_s4.ndim == 2:
        Lij_s4 = Lij_s4[np.newaxis]
    spin, nL, nemb_pair = Lij_s4.shape
    
    if Lkl_s4.ndim == 2:
        Lkl_s4 = Lkl_s4[np.newaxis]

    if spin == 1:
        lib.dot(Lij_s4[0].T, Lkl_s4[0], 1, eri[0], 1) # PL, LQ -> PQ
    else:
        lib.dot(Lij_s4[0].T, Lkl_s4[0], 1, eri[0], 1) # aaaa
        lib.dot(Lij_s4[0].T, Lkl_s4[1], 1, eri[1], 1) # aabb
        lib.dot(Lij_s4[1].T, Lkl_s4[1], 1, eri[2], 1) # bbbb

# *****************************************************************************
# MDF ERI construction
# *****************************************************************************

def get_emb_eri_fast_mdf(cell, mydf, C_ao_lo=None, basis=None, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, unit_eri=False, swap_idx=None, \
        t_reversal_symm=True):
    """
    Fast routine to compute embedding space ERI on the fly, by MDF
    C_ao_lo: (spin, nkpts, nao, nlo), transform matrix from AO to LO basis in k-space
    basis: (spin, ncells, nlo, nemb), embedding basis
    """
    log.info("Get ERI from MDF")

    # MDF variables
    nao = cell.nao_nr()
    kpts = mydf.kpts
    nkpts = len(kpts)
    if mydf.auxcell is None:
        naux = get_naoaux(mydf)
    else:
        naux = mydf.auxcell.nao_nr()
    
    # If C_ao_lo and basis not given, this routine is k2gamma AO transformation
    if C_ao_lo is None:
        C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        C_ao_lo[:, range(nao), range(nao)] = 1.0 # identity matrix for each k
    
    # add spin dimension for restricted C_ao_lo
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    
    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    
    # basis related
    if basis is None:
        basis = np.eye(nao*nkpts).reshape(1, nkpts, nao, nao*nkpts)
    if basis.shape[0] < C_ao_lo.shape[0]:
        basis = add_spin_dim(basis, C_ao_lo.shape[0])
    if C_ao_lo.shape[0] < basis.shape[0]:
        C_ao_lo = add_spin_dim(C_ao_lo, basis.shape[0])
    spin, _, _, nemb = basis.shape
    nemb_pair = nemb*(nemb+1) // 2
    
    if unit_eri:
        C_ao_emb = C_ao_lo / (nkpts**0.75)
    else:
        phase = get_phase_R2k(cell, kpts)
        C_ao_emb = multiply_basis(C_ao_lo, get_basis_k(basis, phase)) / (nkpts**(0.75)) # AO to embedding basis

    # ERI construction
    eri = np.zeros((spin*(spin+1)//2, nemb_pair, nemb_pair), dtype=np.complex128) # 4-fold symmetry
    
    if max_memory is None:
        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0]) * 0.9)
    
    blksize = max_memory * 1e6 / 16 / (nao**2 * 2)
    blksize = max(16, min(int(blksize), mydf.blockdim))
    Lij_s4 = np.empty((spin, naux, nemb_pair), dtype=np.complex128)
    buf = np.empty((spin * blksize * nemb_pair,), dtype=np.complex128)
    
    # GDF part
    for kL in range(nkpts):
        Lij_s4[:] = 0.0
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = max_abs(np.round(kconserv) - kconserv) < kconserv_tol 
                if is_kconserv:
                    log.debug(2, "kL: %s / %s, ki: %s, kj: %s", kL + 1, nkpts, i, j)
                    log.debug(2, "Read Lpq and ao2emb")
                    # Loop over L chunks
                    step0, step1 = 0, 0
                    for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], \
                            max_memory=max_memory, compact=False, \
                            blksize=blksize):
                        Lpq = np.empty(LpqR.shape, dtype=np.complex128)
                        Lpq.real, Lpq.imag = LpqR, LpqI
                        LpqR = LpqI = None
                        lchunk = Lpq.shape[0]
                        step0, step1 = step1, step1 + lchunk
                        
                        Lpq_beta = None
                        Lij_loc = transform_ao_to_emb(Lpq, C_ao_emb, i, j,\
                                Lpq_beta=Lpq_beta)
                        lib.pack_tril(Lij_loc.reshape(-1, nemb, nemb), out=buf)
                        Lij_s4[:, step0:step1] += buf[:(spin * lchunk * \
                                nemb_pair)].reshape(spin, lchunk, nemb_pair)
                        
                    Lpq = Lpq_beta = Lij_loc = None
                    if step1 != naux:
                        log.warn("step1 (%s) != naux (%s), "
                        "auxbasis drop may happened...", step1, naux)
        log.debug(1, "ERI contraction for kL: %5s / %-5s ", kL + 1, nkpts)
        _Lij_s4_to_eri(Lij_s4, eri)

    Lij_s4 = buf = None
    
    # AFT part
    kptij_lst = np.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:,0]
    kptjs_lst = kptij_lst[:,1]
    kpt_ji = kptjs_lst - kptis_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    ngrids = np.prod(mydf.mesh)
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    for uniq_id, kpt in enumerate(uniq_kpts):
        log.debug(1, "q: %s / %s", uniq_id+1, len(uniq_kpts))
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = np.where(uniq_inverse == uniq_id)[0]
        kptjs = kptjs_lst[adapted_ji_idx]
        coulG = mydf.weighted_coulG(q, False, mydf.mesh)
        #coulG *= factor

        # IJ part 
        log.debug(2, "Calculate zij")
        zij_R = np.zeros((spin, ngrids, nemb*nemb), dtype=np.complex128)
        moij_list = []
        ijslice_list = []
        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            moij, ijslice = zip(*[_conc_mos(C_ao_emb[s][ki], C_ao_emb[s][kj])[2:] for s in range(spin)])
            moij_list.append(moij)
            ijslice_list.append(ijslice)
        
        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, kptjs, max_memory=max_memory):
            for ji, aoao in enumerate(aoaoks):
                ki = adapted_ji_idx[ji] // nkpts
                kj = adapted_ji_idx[ji] %  nkpts
                
                buf = aoao.transpose(1,2,0).reshape(nao**2, p1-p0)
                zij = transform_ao_to_emb_aft(lib.transpose(buf), C_ao_emb, \
                        moij_list[ji], ijslice_list[ji])
                zij *= coulG[p0:p1,None]
                zij_R[:, p0:p1] += zij
        
        coulG = buf = aoao = zij = None
        zij_R = _pack_tril(zij_R.reshape(spin, -1, nemb, nemb))
        
        # KL part
        log.debug(2, "Calculate zkl")
        zkl_R = np.zeros((spin, ngrids, nemb*nemb), dtype=np.complex128)
        mokl_list = []
        klslice_list = []
        for kk in range(nkpts):
            kl = kconserv[ki, kj, kk]
            mokl, klslice = zip(*[_conc_mos(C_ao_emb[s][kk], C_ao_emb[s][kl])[2:] for s in range(spin)])
            mokl_list.append(mokl)
            klslice_list.append(klslice)

        ki = adapted_ji_idx[0] // nkpts
        kj = adapted_ji_idx[0] % nkpts
        kptls = kpts[kconserv[ki, kj, :]]
        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, -kptls, max_memory=max_memory):
            for kk, aoao in enumerate(aoaoks):
                buf = aoao.conj().transpose(1,2,0).reshape(nao**2,p1-p0)
                zkl = transform_ao_to_emb_aft(lib.transpose(buf), C_ao_emb, \
                        mokl_list[kk], klslice_list[kk])
                zkl_R[:, p0:p1] += zkl
        
        buf = aoao = zkl = None
        zkl_R = _pack_tril(zkl_R.reshape(spin, -1, nemb, nemb))

        log.debug(1, "Contract ERI")
        _Lij_s4_to_eri_aft(zij_R, zkl_R, eri)
    
    zij_R = zkl_R = None
        
    eri_imag_norm = max_abs(eri.imag)
    log.info('ERI imaginary = %s', eri_imag_norm)
    if eri_imag_norm > ERI_IMAG_TOL:
        log.warn("ERI has imaginary part > %s (%s)", ERI_IMAG_TOL, \
                eri_imag_norm)
    eri = eri.real

    log.debug(1, "ERI restore")
    eri = eri_restore(eri, symmetry, nemb)
    return eri

def get_unit_eri_fast_mdf(cell, mydf, C_ao_lo, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, swap_idx=None, t_reversal_symm=True):
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    basis = np.empty_like(C_ao_lo)
    return get_emb_eri_fast_mdf(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, feri=feri, \
        kscaled_center=kscaled_center, symmetry=symmetry, max_memory=max_memory, \
        kconserv_tol=kconserv_tol, unit_eri=True, swap_idx=swap_idx, \
        t_reversal_symm=t_reversal_symm)

# *****************************************************************************
# FFTDF ERI construction
# *****************************************************************************

def get_emb_eri_fast_fft(cell, mydf, C_ao_lo=None, basis=None, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, unit_eri=False, swap_idx=None, \
        t_reversal_symm=True):
    """
    Fast routine to compute embedding space ERI on the fly, by FFTDF
    C_ao_lo: (spin, nkpts, nao, nlo), transform matrix from AO to LO in k-space
    basis: (spin, ncells, nlo, nemb), embedding basis
    """
    log.info("Get ERI from FFTDF")

    # FFTDF variables
    nao = cell.nao_nr()
    kpts = mydf.kpts
    nkpts = len(kpts)
    
    # If C_ao_lo and basis not given, this routine is k2gamma AO transformation
    if C_ao_lo is None:
        C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        C_ao_lo[:, range(nao), range(nao)] = 1.0 # identity matrix for each k
    
    # add spin dimension for restricted C_ao_lo
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    
    # possible kpts shift
    kscaled = cell.get_scaled_kpts(kpts)
    if kscaled_center is not None:
        kscaled -= kscaled_center
    
    # basis related
    if basis is None:
        basis = np.eye(nao*nkpts).reshape(1, nkpts, nao, nao*nkpts)
    if basis.shape[0] < C_ao_lo.shape[0]:
        basis = add_spin_dim(basis, C_ao_lo.shape[0])
    if C_ao_lo.shape[0] < basis.shape[0]:
        C_ao_lo = add_spin_dim(C_ao_lo, basis.shape[0])
    spin, _, _, nemb = basis.shape
    nemb_pair = nemb*(nemb+1) // 2
    
    if unit_eri:
        C_ao_emb = C_ao_lo / (nkpts**0.75)
    else:
        phase = get_phase_R2k(cell, kpts)
        C_ao_emb = multiply_basis(C_ao_lo, get_basis_k(basis, phase)) / (nkpts**0.75) # AO to embedding basis
    
    # C_ao_emb in real grids, shape (spin, nkpts, nemb, ngrids)
    coords = cell.gen_uniform_grids(mydf.mesh)
    aos = np.asarray(mydf._numint.eval_ao(cell, coords, kpts))
    moT = np.empty((spin, nkpts, nemb, aos.shape[-2]), dtype=np.complex128)
    for s in range(spin):
        for k in range(nkpts):
            lib.dot(C_ao_emb[s,k].T, aos[k].T, 1, moT[s,k])

    # ERI construction
    eri = np.zeros((spin*(spin+1)//2, nemb_pair, nemb_pair), dtype=np.complex128) # 4-fold symmetry
   
    # FFT variables
    kptij_lst = np.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:,0]
    kptjs_lst = kptij_lst[:,1]
    kpt_ji = kptjs_lst - kptis_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    ngrids = np.prod(mydf.mesh)
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    for uniq_id, kpt in enumerate(uniq_kpts):
        log.debug(1, "q: %s / %s", uniq_id+1, len(uniq_kpts))
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = np.where(uniq_inverse == uniq_id)[0]

        coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
        coulG *= (cell.vol/ngrids)
        phase = np.exp(1j * np.dot(coords, q))
        
        log.debug(2, "Calculate zij")
        zij_R = 0.0
        for ji_idx in adapted_ji_idx:
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            zij_R += get_mo_pairs(moT, ki, kj, phase=phase)

        zij_R = tools.fft(zij_R.reshape(-1,ngrids), mydf.mesh) * coulG
        zij_R = tools.ifft(zij_R.reshape(-1,ngrids), mydf.mesh) * phase
        zij_R = _pack_tril(zij_R.reshape(spin, nemb, nemb, ngrids).transpose(0, 3, 1, 2))
        coulG = phase = None
        
        log.debug(2, "Calculate zkl")
        ki = adapted_ji_idx[0] // nkpts
        kj = adapted_ji_idx[0] % nkpts
        zkl_R = 0.0
        for kk in range(nkpts):
            kl = kconserv[ki, kj, kk]
            zkl_R += get_mo_pairs(moT, kk, kl, phase=None)
        
        zkl_R = _pack_tril(zkl_R.transpose(0, 3, 1, 2))
        
        log.debug(1, "Contract ERI")
        _Lij_s4_to_eri_aft(zij_R, zkl_R, eri)
    
    zij_R = zkl_R = None
        
    eri_imag_norm = max_abs(eri.imag)
    log.info('ERI imaginary = %s', eri_imag_norm)
    if eri_imag_norm > ERI_IMAG_TOL:
        log.warn("ERI has imaginary part > %s (%s)", ERI_IMAG_TOL, \
                eri_imag_norm)
    eri = eri.real

    log.debug(1, "ERI restore")
    eri = eri_restore(eri, symmetry, nemb)
    return eri

def get_unit_eri_fast_fft(cell, mydf, C_ao_lo, feri=None, \
        kscaled_center=None, symmetry=1, max_memory=None, \
        kconserv_tol=KPT_DIFF_TOL, swap_idx=None, t_reversal_symm=True):
    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    basis = np.empty_like(C_ao_lo)
    return get_emb_eri_fast_fft(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, feri=feri, \
        kscaled_center=kscaled_center, symmetry=symmetry, max_memory=max_memory, \
        kconserv_tol=kconserv_tol, unit_eri=True, swap_idx=swap_idx, \
        t_reversal_symm=t_reversal_symm)

def get_mo_pairs(moT, kk, kl, phase=None):
    spin, nkpts, nemb, ngrids = moT.shape
    mo_pairs = np.empty((spin, nemb, nemb, ngrids), dtype=np.complex128)
    for s in range(spin):
        if phase is None:
            #mo_pairs[s] = np.einsum('ig, jg -> ijg', moT[s, kk].conj(), moT[s, kl])
            np.multiply(moT[s,kk].conj()[:, None], moT[s,kl], out=mo_pairs[s])
        else:
            #mo_pairs[s] = np.einsum('ig, g, jg -> ijg', moT[s, kk].conj(), phase.conj(), moT[s, kl])
            np.multiply((moT[s, kk]*phase).conj()[:, None], moT[s,kl], out=mo_pairs[s])
    return mo_pairs

# *****************************************************************************
# Test functions:
# *****************************************************************************

def transform_gdf_to_lo(mydf, C_ao_lo, fname="gdf_ints_lo.h5", \
        t_reversal_symm=True):
    """
    Transform the gdf integral to LO basis.
    
    Args:
        mydf: gdf object.
        C_ao_lo: LO coefficients, shape (nkpts, nao, nlo).
        fname: filename to save the new integral.
        t_reversal_symm: use time reversal symmetry.

    Returns:
        mydf_lo: a new df object.
    """
    log.info("Transform GDF to LO.")
    cell = mydf.cell
    kpts = mydf.kpts
    max_memory = mydf.max_memory
    nkpts, nao, nlo = C_ao_lo.shape 
    assert nkpts == len(kpts)
    assert nao == cell.nao_nr()
    assert mydf._cderi is not None
    if mydf.auxcell is None:
        naux = get_naoaux(mydf)
    else:
        naux = mydf.auxcell.nao_nr()
    
    dataname = 'j3c'
    feri = h5py.File(mydf._cderi, 'r')
    feri_lo = h5py.File(fname, 'w')
    kptij_lst = np.asarray(feri[dataname + '-kptij'])
    nkptij = len(kptij_lst)
    feri_lo[dataname + '-kptij'] = kptij_lst

    if t_reversal_symm:
        mask = get_mask_kptij_lst(cell, kptij_lst, tol=KPT_DIFF_TOL)
    else:
        mask = -np.ones(len(kptij_lst), dtype=np.int)
    
    Lij = np.empty((naux, nlo * nlo), dtype=np.complex128)
    
    # ZHC TODO: slice the LO indices, not L.
    for k in range(nkptij):
        log.debug(2, "kpt pair: %5s / %-5s", k, nkptij)
        if mask[k] == -2: # already computed
            continue
        else:
            kpti, kptj = kptij_lst[k]
            is_real = gamma_point(kptij_lst[k])
            aosym_ks2 = gamma_point(kpti - kptj)
            i, j = member(kpti, kpts)[0], member(kptj, kpts)[0]
            L_id = 0
            step0, step1 = 0, 0
            Lij[:] = 0.0 # avoid aux basis is not equal size.
            for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], \
                    max_memory=max_memory, compact=False):
                Lpq = np.empty(LpqR.shape, dtype=np.complex128)
                Lpq.real, Lpq.imag = LpqR, LpqI
                LpqR = LpqI = None
                step0, step1 = step1, step1 + Lpq.shape[0]
                Lij[step0:step1] = transform_ao_to_emb(Lpq, \
                        C_ao_lo, i, j)[0]
            if step1 != naux:
                log.warn("step1 (%s) != naux (%s), " \
                        "auxbasis drop may happened...", step1, naux)
            
            if is_real:
                assert max_abs(Lij.imag) < ERI_IMAG_TOL
                Lij_s4 = lib.pack_tril(Lij.real.reshape(-1, nlo, nlo))
            elif aosym_ks2:
                Lij_s4 = lib.pack_tril(Lij.reshape(-1, nlo, nlo))
            else:
                Lij_s4 = Lij

            feri_lo['%s/%d/%d' % (dataname, k, L_id)] = Lij_s4
            if mask[k] != -1: # t_reversal_symm
                feri_lo['%s/%d/%d' % (dataname, mask[k], L_id)] = Lij_s4.conj()
        Lpq = None
    feri.close()
    feri_lo.close()
    if nlo != nao:
        # new gdf_lo object, cell.nao_nr is rewritten
        cell = cell.copy()
        cell.nao_nr = lambda *args: nlo
    mydf_lo = mydf.__class__(cell, kpts)
    mydf_lo._cderi = fname
    return mydf_lo

def get_mask_kptij_lst(cell, kptij_lst, tol=KPT_DIFF_TOL):
    """
    kpts pair mask for time reversal symmetry.
    Note: -1 for self map, -2 for already used.
    """
    nkpts_pair = len(kptij_lst)
    kptij_scaled = cell.get_scaled_kpts(kptij_lst)
    kptij_round = round_to_FBZ(kptij_scaled, tol=tol)
    mask = -np.ones(nkpts_pair, dtype=np.int)
    for i, ki in enumerate(kptij_round):
        if mask[i] == -1:
            for j in range(i + 1, nkpts_pair):
                sum_ij = ki + kptij_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    mask[i] = j
                    mask[j] = -2
                    break
    return mask

if __name__ == '__main__':
    import os
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import df
    import pyscf.pbc.scf as pscf
    from pyscf.pbc.lib import chkfile
    from libdmet_solid.system import lattice
    import libdmet_solid.lo.pywannier90 as pywannier90
    from libdmet_solid.utils.misc import mdot
    np.set_printoptions(3, linewidth=1000)

    cell = pgto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
    
    # Latice class
    nk = kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts_abs = Lat.kpts_abs
    nkpts = Lat.nkpts

    chkfname = 'hchain.chk'
    if os.path.isfile(chkfname):
    #if False:
        kmf = pscf.KRHF(cell, kpts_abs)
        gdf = df.GDF(cell, kpts_abs)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-12
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = pscf.KRHF(cell, kpts_abs)
        gdf = df.GDF(cell, kpts_abs)
        kmf = kmf.density_fit()
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()


