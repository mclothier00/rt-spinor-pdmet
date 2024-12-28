#! /usr/bin/env python

"""
Helper functions for pyscf PBC module.
Including KRHF, KUHF, get_eri, get_jk.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import numpy as np

from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
from pyscf import pbc
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.lib import logger as pyscflogger
from pyscf.scf import ghf as mol_ghf

from libdmet_solid.solver import scf as scf_helper
from libdmet_solid.utils.misc import mdot, max_abs, add_spin_dim
from libdmet_solid.settings import IMAG_DISCARD_TOL 
from libdmet_solid.utils import logger as log
from libdmet_solid.routine.pdft_helper import *

flush = log.flush_for_pyscf
pyscflogger.flush = flush([""])

# *********************************************************************
# KRHF, KUHF
# get_jk for 7d, local and nearest ERI.
# *********************************************************************

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''
    Following pyscf.scf.hf.energy_elec()
    Support h1e with spin.
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    h1e_kpts = np.asarray(h1e_kpts)
    if h1e_kpts.ndim == 3:
        h1e_kpts = np.asarray((h1e_kpts, h1e_kpts))
    nkpts = h1e_kpts.shape[-3]
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts[0], h1e_kpts[0])
    e1+= 1./nkpts * np.einsum('kij,kji', dm_kpts[1], h1e_kpts[1])
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts[0], vhf_kpts[0]) * 0.5
    e_coul+= 1./nkpts * np.einsum('kij,kji', dm_kpts[1], vhf_kpts[1]) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    pyscflogger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if abs(e_coul.imag > mf.cell.precision*10):
        log.warn("Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

class KUHF(pbc.scf.kuhf.KUHF):
    """
    KUHF that supports hcore with spin.
    """
    def init_guess_by_1e(self, cell=None):
        """
        Support h1e with spin dimension.
        """
        if cell is None: cell = self.cell
        h1e = np.asarray(self.get_hcore(cell))
        s1e = self.get_ovlp(cell)
        if h1e.ndim == 3: 
            mo_energy, mo_coeff = self.eig((h1e, h1e), s1e)
        else:
            mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        dma, dmb = self.make_rdm1(mo_coeff, mo_occ)
        return np.array((dma, dmb))
    
    energy_elec = energy_elec

def get_eri_7d(cell, xdf, kpts=None, compact=False):
    """
    Get eri 7d in ao basis.
    """
    nao = cell.nao_nr()
    if kpts is None:
        kpts = xdf.kpts 
    nkpts = len(kpts)
    eri_7d = np.zeros((nkpts, nkpts, nkpts, nao, nao, nao, nao), \
            dtype=np.complex128)
    kconserv = get_kconserv(cell, kpts)
    for i, kpti in enumerate(kpts):
        for j, kptj in enumerate(kpts):
            for k, kptk in enumerate(kpts):
                l = kconserv[i, j, k]
                kptl = kpts[l]
                eri_7d[i, j, k] = xdf.get_eri((kpti, kptj, kptk, kptl), \
                        compact=compact).reshape((nao,)*4) 
    return eri_7d

def get_jk_from_eri_7d(eri, dm, with_j=True, with_k=True):
    """
    Get J, K matrix in kpts. 
    Assume eri is spinless, 7d.
    """
    eri = np.asarray(eri)
    dm = np.asarray(dm)
    old_shape = dm.shape
    if dm.ndim == 3:
        dm = dm[None]
    spin, nkpts, nao, _ = dm.shape
    
    if with_j:
        vj = np.zeros((spin, nkpts, nao, nao), \
                dtype=np.result_type(dm.dtype, eri.dtype))
        if with_k: # J and K
            vk = np.zeros((spin, nkpts, nao, nao), \
                    dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                for k in range(nkpts):
                    vj[s] += lib.einsum('Rpqrs, qp -> Rrs', eri[k, k], dm[s, k])
                    vk[s] += lib.einsum('Ppqrs, qr -> Pps', eri[:, k, k], dm[s, k])
            vj /= float(nkpts)
            vk /= float(nkpts)
            vj = vj.reshape(old_shape)
            vk = vk.reshape(old_shape)
        else: # only J
            vk = None
            for s in range(spin):
                for k in range(nkpts):
                    vj[s] += lib.einsum('Rpqrs, qp -> Rrs', eri[k, k], dm[s, k])
            vj /= float(nkpts)
            vj = vj.reshape(old_shape)
    else:
        if with_k: # only K
            vj = None
            vk = np.zeros((spin, nkpts, nao, nao), \
                    dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                for k in range(nkpts):
                    vk[s] += lib.einsum('Ppqrs, qr -> Pps', eri[:, k, k], dm[s, k])
            vk /= float(nkpts)
            vk = vk.reshape(old_shape)
        else: # no J no K
            vj = vk = None
    return vj, vk

def get_j_from_eri_7d(eri, dm):
    """
    Get J matrix in kpts. 
    Assume eri is spinless, 7d.
    """
    return get_jk_from_eri_7d(eri, dm, with_j=True, with_k=False)[0]

def get_k_from_eri_7d(eri, dm):
    """
    Get K matrix in kpts. 
    Assume eri is spinless, 7d.
    """
    return get_jk_from_eri_7d(eri, dm, with_j=False, with_k=True)[1]

def get_jk_from_eri_local(eri, dm, eri_symm=4, with_j=True, with_k=True):
    """
    Get J, K matrix in kpts. 
    Assume eri is local, spinless.
    """
    dm = np.asarray(dm)
    old_shape = dm.shape
    if dm.ndim == 3:
        dm = dm[None]
    spin, nkpts, nao, _ = dm.shape
    if eri.size == nao**4:
        eri = ao2mo.restore(eri_symm, eri, nao)
    
    dm_ave = dm.sum(axis=-3) / float(nkpts)
    if max_abs(dm_ave.imag) < IMAG_DISCARD_TOL:
        dm_ave = dm_ave.real
    else:
        log.warn("get_*_from_eri_local assume a real dm_ave," + \
                "now imag = %s", max_abs(dm_ave.imag))
        
    if with_j:
        vj = np.zeros((spin, nkpts, nao, nao), \
                dtype=np.result_type(dm.dtype, eri.dtype))
        if with_k: # J and K
            vk = np.zeros((spin, nkpts, nao, nao), \
                    dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                vj[s], vk[s] = scf.hf.dot_eri_dm(eri, dm_ave[s], hermi=1, \
                        with_j=with_j, with_k=with_k)
            vj = vj.reshape(old_shape)
            vk = vk.reshape(old_shape)
        else: # only J
            vk = None
            for s in range(spin):
                vj[s], _ = scf.hf.dot_eri_dm(eri, dm_ave[s], hermi=1, \
                        with_j=with_j, with_k=with_k)
            vj = vj.reshape(old_shape)
    else:
        if with_k: # only K
            vj = None
            vk = np.zeros((spin, nkpts, nao, nao), \
                    dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                _, vk[s] = scf.hf.dot_eri_dm(eri, dm_ave[s], hermi=1, \
                        with_j=with_j, with_k=with_k)
            vk = vk.reshape(old_shape)
        else: # no J no K
            vj = vk = None
    return vj, vk

def get_j_from_eri_local(eri, dm, eri_symm=4):
    """
    Get J matrix in kpts. 
    Assume eri is local, spinless.
    """
    return get_jk_from_eri_local(eri, dm, eri_symm=eri_symm, \
            with_j=True, with_k=False)[0]

def get_k_from_eri_local(eri, dm, eri_symm=4):
    """
    Get K matrix in kpts. 
    Assume eri is local, spinless.
    """
    return get_jk_from_eri_local(eri, dm, eri_symm=eri_symm, \
            with_j=False, with_k=True)[1]

def get_jk_from_eri_nearest(eri, dm, lattice, with_j=True, with_k=True):
    """
    Get J, K matrix in kpts. 
    Assume eri is nearest, spinless, in real space representation,
    shape (nR, nao, nao, nao, nao)
    dm is in kspace.
    """
    dm = np.asarray(dm)
    old_shape = dm.shape
    if dm.ndim == 3:
        dm = dm[None]
    spin, nkpts, nao, _ = dm.shape
    
    dm_R = lattice.k2R(dm)
    if max_abs(dm_R.imag) < IMAG_DISCARD_TOL:
        dm_R = dm_R.real
    else:
        log.warn("get_*_from_eri_nearest assume a real dm_R," + \
                "now imag = %s", max_abs(dm_R.imag))
    eri_ave = eri.sum(axis=-5)

    if with_j:
        vj = np.zeros((spin, nkpts, nao, nao), dtype=np.complex128)
        if with_k: # J and K
            vk_R = np.zeros((spin, nkpts, nao, nao))
            for s in range(spin):
                vj[s] = lib.einsum('qp, pqrs -> rs', dm_R[s, 0], eri_ave)
                for R in range(nkpts):
                    vk_R[s, R] = lib.einsum('pqrs, qr -> ps', eri[R], dm_R[s, R])
            vk = lattice.R2k(vk_R)
            vj = vj.reshape(old_shape)
            vk = vk.reshape(old_shape)
        else: # only J
            vk = None
            for s in range(spin):
                vj[s] = lib.einsum('qp, pqrs -> rs', dm_R[s, 0], eri_ave)
            vj = vj.reshape(old_shape)
    else:
        if with_k: # only K
            vj = None
            vk_R = np.zeros((spin, nkpts, nao, nao))
            for s in range(spin):
                vj[s] = lib.einsum('qp, pqrs -> rs', dm_R[s, 0], eri_ave)
                for R in range(nkpts):
                    vk_R[s, R] = lib.einsum('pqrs, qr -> ps', eri[R], dm_R[s, R])
            vk = lattice.R2k(vk_R)
            vk = vk.reshape(old_shape)
        else: # no J no K
            vj = vk = None
    return vj, vk

def get_j_from_eri_nearest(eri, dm, lattice):
    """
    Get J matrix in kpts. 
    Assume eri is nearest, spinless.
    """
    return get_jk_from_eri_nearest(eri, dm, lattice, with_j=True, \
            with_k=False)[0]

def get_k_from_eri_nearest(eri, dm, lattice):
    """
    Get K matrix in kpts. 
    Assume eri is nearest, spinless.
    """
    return get_jk_from_eri_nearest(eri, dm, lattice, with_j=False, \
            with_k=True)[1]

def smearing_(mf, sigma=None, method='fermi', mu0=None, tol=1e-12):
    """
    Fermi-Dirac or Gaussian smearing.
    This version support Sz for UHF smearing.
    """
    from pyscf.scf import uhf
    from pyscf.scf import ghf
    from pyscf.pbc.scf import khf
    from libdmet_solid.routine import mfd, ftsystem
    mf_class = mf.__class__
    is_uhf = isinstance(mf, uhf.UHF)
    is_ghf = isinstance(mf, ghf.GHF)
    is_rhf = (not is_uhf) and (not is_ghf)
    is_khf = isinstance(mf, khf.KSCF)
    if hasattr(mf, "cell"):
        Sz = mf.cell.spin
    else:
        Sz = mf.mol.spin
    tol = min(mf.conv_tol * 0.01, tol)

    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mo_occ_kpts

        if is_khf:
            nkpts = len(mf.kpts)
        else:
            nkpts = 1
        
        # find nelec_target
        nelectron = mf.cell.tot_electrons(nkpts)
        if is_uhf:
            nelec_target = [(nelectron + Sz) * 0.5, (nelectron - Sz) * 0.5]
        elif is_ghf:
            nelec_target = nelectron
        else:
            nelec_target = nelectron * 0.5
        
        if mf.smearing_method.lower() == 'fermi':  # Fermi-Dirac smearing
            f_occ = ftsystem.fermi_smearing_occ
        else:  # Gaussian smearing
            f_occ = ftsystem.gaussian_smearing_occ
        
        mo_energy = np.asarray(mo_energy_kpts)
        mo_occ, mf.mu, nerr = mfd.assignocc(mo_energy, nelec_target, 1.0/mf.sigma, mf.mu, \
                fit_tol=tol, f_occ=f_occ)
        mo_occ = mo_occ.reshape(mo_energy.shape)

        # See https://www.vasp.at/vasp-workshop/slides/k-points.pdf
        if mf.smearing_method.lower() == 'fermi':
            f = mo_occ[(mo_occ>0) & (mo_occ<1)]
            mf.entropy = -(f*np.log(f) + (1-f)*np.log(1-f)).sum() / nkpts
        else:
            if is_uhf:
                mf.entropy = (np.exp(-((mo_energy[0]-mf.mu[0])/mf.sigma)**2).sum()
                            / (2*np.sqrt(np.pi)) / nkpts) + \
                             (np.exp(-((mo_energy[1]-mf.mu[1])/mf.sigma)**2).sum()
                            / (2*np.sqrt(np.pi)) / nkpts)
            else:
                mf.entropy = (np.exp(-((mo_energy-mf.mu)/mf.sigma)**2).sum()
                            / (2*np.sqrt(np.pi)) / nkpts)
        if is_rhf:
            mo_occ *= 2
            mf.entropy *= 2
        
        nelec_now = mo_occ.sum()
        logger.debug(mf, '    Fermi level %s  Sum mo_occ_kpts = %s  should equal nelec = %s',
                     mf.mu, nelec_now, nelectron)
        if abs(nelec_now - nelectron) > 1e-8:
            log.warn("Occupancy (nelec_now %s) is not equal to cell.nelectron (%s).", \
                    nelec_now, nelectron)
        logger.info(mf, '    sigma = %g  Optimized mu = %s  entropy = %.12g',
                    mf.sigma, mf.mu, mf.entropy)
        return mo_occ

    def get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
        if is_khf:
            grad_kpts = []
            for k, mo in enumerate(mo_coeff_kpts):
                f_mo = mdot(mo.T.conj(), fock[k], mo)
                nmo = f_mo.shape[0]
                grad_kpts.append(f_mo[np.tril_indices(nmo, -1)])
            return np.hstack(grad_kpts)
        else:
            f_mo = mdot(mo_coeff_kpts.T.conj(), fock, mo_coeff_kpts)
            nmo = f_mo.shape[0]
            return f_mo[np.tril_indices(nmo, -1)]

    def get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None):
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mf_class.get_grad(mf, mo_coeff_kpts, mo_occ_kpts, fock)
        if fock is None:
            dm1 = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = mf.get_hcore() + mf.get_veff(mf.cell, dm1)
        if is_uhf:
            ga = get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return np.hstack((ga,gb))
        else: # rhf and ghf
            return get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

    def energy_tot(dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        e_tot = mf.energy_elec(dm_kpts, h1e_kpts, vhf_kpts)[0] + mf.energy_nuc()
        if (mf.sigma and mf.smearing_method and
            mf.entropy is not None):
            mf.e_free = e_tot - mf.sigma * mf.entropy
            mf.e_zero = e_tot - mf.sigma * mf.entropy * .5
            logger.info(mf, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, mf.e_free, mf.e_zero)
        return e_tot

    mf.sigma = sigma
    mf.smearing_method = method
    mf.entropy = None
    mf.e_free = None
    mf.e_zero = None
    if mu0 is None:
        mf.mu = 0.0
    else:
        mf.mu = mu0
    mf._keys = mf._keys.union(['sigma', 'smearing_method',
                               'entropy', 'e_free', 'e_zero', 'mu'])

    mf.get_occ = get_occ
    mf.energy_tot = energy_tot
    mf.get_grad = get_grad
    return mf


