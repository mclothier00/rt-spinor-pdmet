#! /usr/bin/env python

"""
CC impurity solver.

Author:
    Zhi-Hao Cui
"""

import os
import numpy as np
import scipy.linalg as la
import h5py
import time

from libdmet_solid.utils import logger as log
from libdmet_solid.solver import scf
from libdmet_solid.solver.scf import ao2mo_Ham, restore_Ham
from libdmet_solid.basis_transform.make_basis import \
        transform_rdm1_to_ao_mol, transform_rdm2_to_ao_mol, rotate_emb_basis, \
        find_closest_mo, trans_mo, get_mo_ovlp
from libdmet_solid.utils.misc import mdot, max_abs

from pyscf import cc
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc.uccsd import _ChemistsERIs
from pyscf.cc.gccsd import _PhysicistsERIs

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    """
    Hacked CC make eri function. NOTE the order.
    """
    cput0 = (time.clock(), time.time())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    else:
        if len(mycc._scf._eri) == 1:
            eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], moa), nmoa)
            eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], mob), nmob)
            eri_ab = ao2mo.general(mycc._scf._eri[0], (moa, moa, mob, mob), \
                    compact=False)
        elif len(mycc._scf._eri) == 3:
            # ZHC NOTE the order, aa, bb, ab
            eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], moa), nmoa)
            eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[1], mob), nmob)
            eri_ab = ao2mo.general(mycc._scf._eri[2], (moa, moa, mob, mob), \
                    compact=False)
        else:
            raise ValueError

    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    if not callable(ao2mofn):
        ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
        vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
        idxa = np.tril_indices(nvira)
        idxb = np.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

        OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    return eris

class UICCSD(cc.uccsd.UCCSD):
    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.get_nmo()
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmoa * (nmoa+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'UCCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            raise NotImplementedError

        else:
            raise NotImplementedError
            return _make_eris_outcore(self, mo_coeff)

def _make_eris_incore_ghf(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.clock(), time.time())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        assert eris.mo_coeff.dtype == np.double
        eri = ao2mo.kernel(mycc._scf._eri, eris.mo_coeff) 
        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)

    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
    return eris

class GGCCSD(cc.gccsd.GCCSD):
    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        mem_incore = nmo**4*2 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore_ghf(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError
            return _make_eris_outcore(self, mo_coeff)

class CCSD(object):
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, \
            restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-9, \
            tol_normt=1e-6, max_cycle=200, level_shift=0.0, frozen=0, \
            max_memory=40000, compact_rdm2=False, scf_newton=True):
        """
        CCSD solver.
        """
        self.restricted = restricted
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = tol
        self.conv_tol_normt = tol_normt
        self.level_shift = level_shift
        self.frozen = frozen
        self.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.Sz = Sz
        self.scfsolver = scf.SCF(newton_ah=scf_newton)
        
        self.t12 = None
        self.l12 = None
        self.onepdm = None
        self.twopdm = None
        self.compact_rdm2 = compact_rdm2 # ZHC TODO symm of rdm2

        self.optimized = False
    
    def run(self, Ham=None, nelec=None, guess=None, restart=False, \
            dump_tl=False, fcc_name="fcc.h5", calc_rdm2=False, \
            **kwargs):
        """
        Main function of the solver.
        NOTE: the spin order for H2 is aa, bb, ab.
        """
        # 1. sanity check
        log.info("CC solver Run")
        # spin
        spin = Ham.H1["cd"].shape[0]
        if spin > 1: # must be UHF
            assert not self.restricted
        # nelec
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            elif self.ghf:
                nelec = Ham.norb // 2
            else:
                raise ValueError
        nelec_a = (nelec + self.Sz) // 2
        nelec_b = (nelec - self.Sz) // 2
        assert (nelec_a >= 0) and (nelec_b >=0) and (nelec_a + nelec_b == nelec)
        
        # 2. mean-field calculation
        log.debug(1, "CC solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        self.scfsolver.set_system(nelec, self.Sz, False, \
                self.restricted, max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)
        
        # ZHC TODO simplify the logic of restart.
        bcc = kwargs.get("bcc", False) # brueckner CC
        bcc_verbose = kwargs.get("bcc_verbose", 2)
        bcc_restart = kwargs.get("bcc_restart", False)
        if bcc and bcc_restart and self.optimized and restart:
            bcc_restart = True
            scf_max_cycle = 1 # need not to do scf
        else:
            bcc_restart = False
        
        if "mo_coeff_custom" in kwargs:
            # use customized MO as CC reference
            log.info("Use customized MO as CC reference.")
            self.scfsolver.mf.mo_energy = kwargs["mo_energy_custom"]        
            self.scfsolver.mf.mo_coeff = kwargs["mo_coeff_custom"]        
            self.scfsolver.mf.mo_occ = kwargs["mo_occ_custom"]        
            self.scfsolver.mf.e_tot = self.scfsolver.mf.energy_tot()
        else:
            if self.ghf:
                E_HF, rhoHF = self.scfsolver.GGHF(tol=self.conv_tol*0.1, \
                        MaxIter=scf_max_cycle, InitGuess=dm0)
            else:
                E_HF, rhoHF = self.scfsolver.HF(tol=self.conv_tol*0.1, \
                        MaxIter=scf_max_cycle, InitGuess=dm0)
            log.debug(1, "CC solver: mean-field converged: %s", \
                    self.scfsolver.mf.converged)
        
        log.debug(2, "CC solver: mean-field rdm1: \n%s", \
                self.scfsolver.mf.make_rdm1())
        
        # 3. restart can use the previously saved t1 and t2 
        if restart:
            log.eassert("basis" in kwargs, "restart requires basis passed in")
        if restart and self.optimized:
            t1, t2 = self.load_t12_from_h5(fcc_name, kwargs["basis"],
                    self.scfsolver.mf.mo_coeff, bcc_restart=bcc_restart)
            l1, l2 = None, None
        else:
            if guess is not None:
                if len(guess) == 2:
                    t1, t2 = guess
                    l1, l2 = None, None
                else:
                    t1, t2, l1, l2 = guess
            else:
                t1, t2, l1, l2 = None, None, None, None

        # 4. then do CC
        if self.ghf:
            self.cisolver = GGCCSD(self.scfsolver.mf)
        elif Ham.restricted:
            self.cisolver = cc.CCSD(self.scfsolver.mf)
        else:
            self.cisolver = UICCSD(self.scfsolver.mf)
        self.cisolver.max_cycle = self.max_cycle
        self.cisolver.conv_tol = self.conv_tol
        self.cisolver.conv_tol_normt = self.conv_tol_normt
        self.cisolver.level_shift = self.level_shift
        self.cisolver.set(frozen = self.frozen)
        self.cisolver.verbose = self.verbose
        
        # solve t
        log.debug(1, "CC solver: solve t amplitudes")
        E_corr, t1, t2 = self.cisolver.kernel(t1=t1, t2=t2)
        
        # brueckner CC
        if bcc:
            log.info("Using Brueckner CC.")
            self.cisolver = bcc_loop(self.cisolver, utol=self.conv_tol_normt,
                    verbose=bcc_verbose)
            self.scfsolver.mf.mo_coeff = self.cisolver.mo_coeff
            self.scfsolver.mf.e_tot = self.cisolver._scf.e_tot
            t1, t2 = self.cisolver.t1, self.cisolver.t2

        # solve lambda
        log.debug(1, "CC solver: solve l amplitudes")
        l1, l2 = self.cisolver.solve_lambda(t1=t1, t2=t2, l1=l1, l2=l2)
        
        # 6. collect properties
        # ZHC TODO a more compact way to store rdm2 
        # or directly calculate E
        E = self.cisolver.e_tot
        self.make_rdm1(Ham)
        if calc_rdm2:
            self.make_rdm2(Ham)

        log.info("CC solver converged: %s", self.cisolver.converged)
        if not self.cisolver.converged:
            log.warn("CC solver not converged...")
        self.optimized = True

        # dump t1, t2, basis, mo_coeff
        if dump_tl or restart:
            self.save_t12_to_h5(fcc_name, kwargs["basis"],
                    self.cisolver.mo_coeff)
        
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, save_dmet_ham=False, \
            dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False, \
            **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, \
                compact=True, in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        # calculate rdm2 in aa, bb, ab order
        if use_calculated_twopdm:
            log.info("Using exisiting twopdm in MO basis...")
            assert self.twopdm_mo is not None
        else:
            self.make_rdm2(Ham)

        if self.ghf:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            r1 = self.onepdm
            r2 = self.twopdm
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.sum(h1.T * r1)
            E2 = np.sum(h2 * r2) * 0.5
        elif Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.sum(h1[0].T*r1[0]) * 2.0
            E2 = np.sum(h2[0]*r2[0]) * 0.5
        else:
            h1 = Ham.H1["cd"]
            # h2 is in aa, bb, ab order
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            # r2 is in aa, bb, ab order
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            # energy
            E1 = np.tensordot(h1, r1, axes=((0,1,2), (0,2,1)))
            E2_aa = 0.5 * np.sum(r2[0]*h2[0])
            E2_bb = 0.5 * np.sum(r2[1]*h2[1])
            E2_ab = np.sum(r2[2]*h2[2])
            E2 = E2_aa + E2_bb + E2_ab
        
        E = E1 + E2
        E += Ham.H0
        log.debug(0, "run DMET Hamiltonian:\nE0 = %20.12f, E1 = %20.12f, " 
                "E2 = %20.12f, E = %20.12f", Ham.H0, E1, E2, E)
        
        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H0'] = Ham.H0
            fdmet_ham['H1'] = h1
            fdmet_ham['H2'] = h2
            fdmet_ham['mo_coeff'] = self.cisolver.mo_coeff
            fdmet_ham.close()
        return E
    
    def make_rdm1(self, Ham=None):
        log.debug(1, "CC solver: solve rdm1")
        onepdm = self.cisolver.make_rdm1()
        if self.ghf: # GHF
            self.onepdm_mo = np.asarray(onepdm)
        elif self.restricted:
            self.onepdm_mo = np.asarray(onepdm)[np.newaxis] * 0.5
        else:
            self.onepdm_mo = np.asarray(onepdm)

        # rotate back to the AO basis
        log.debug(1, "CC solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, \
                self.cisolver.mo_coeff)
        return self.onepdm

    def make_rdm2(self, Ham=None, ao_repr=False):
        """
        Compute rdm2.
        NOTE: the returned value's spin order for H2 is aa, bb, ab.
        """
        log.debug(1, "CC solver: solve rdm2")
        if self.ghf: # GHF
            self.twopdm_mo = self.cisolver.make_rdm2()
        elif self.restricted:
            self.twopdm_mo = self.cisolver.make_rdm2()[np.newaxis]
        else:
            # NOTE: here is aa, ab, bb order
            self.twopdm_mo = np.asarray(self.cisolver.make_rdm2())

        # rotate back to the AO basis
        # NOTE: the transform function use aa, ab, bb order.
        if ao_repr:
            log.debug(1, "CC solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, \
                    self.cisolver.mo_coeff)
        else:
            self.twopdm = None
            
        if not self.restricted and not self.ghf:
            self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]
        return self.twopdm

    def load_t12_from_h5(self, fcc_name, basis_new, mo_coeff_new, \
            bcc_restart=False):
        """
        Load t1, t2, and rotate to current basis.
        """
        # ZHC FIXME TODO support frozen restart, where the stored mo_coeff
        # should be from eris.
        if not os.path.isfile(fcc_name):
            log.info("CC solver: read previous t and basis failed, "
                    "file %s does not exist.", fcc_name)
            return None, None
        log.debug(1, "CC solver: read previous t and basis")
        fcc = h5py.File(fcc_name, 'r')
        basis_old = np.asarray(fcc['basis'])
        mo_coeff_old = np.asarray(fcc['mo_coeff'])
        if mo_coeff_old.ndim == 2:
            t1_old = np.asarray(fcc['t1'])
            t2_old = np.asarray(fcc['t2'])
        else:
            spin = mo_coeff_old.shape[0]
            t1_old = [np.asarray(fcc['t1_%s'%s]) for s in range(spin)]
            t2_old = [np.asarray(fcc['t2_%s'%s]) for s in \
                    range(spin*(spin+1)//2)]
        fcc.close()
        
        mo_coeff_new = np.asarray(mo_coeff_new)
        assert mo_coeff_new.shape == mo_coeff_old.shape
        nao, nmo = mo_coeff_new.shape[-2:]
        is_same_basis = (basis_new is basis_old) or \
                (max_abs(basis_new - basis_old) < 1e-12)

        if mo_coeff_new.ndim == 2: # RHF and GHF 
            if bcc_restart:
                # restart for a bcc calculation, 
                # new mo is rotated from the old one.
                # umat: C_old_new
                if is_same_basis:
                    mo_coeff_new = mo_coeff_old
                else:
                    basis_old = basis_old.reshape(-1, nmo)
                    basis_new = basis_new.reshape(-1, nmo)
                    umat = find_closest_mo(basis_old, basis_new, \
                            return_rotmat=True)[1]
                    mo_coeff_new = np.dot(umat.conj().T, mo_coeff_old)
            
            if is_same_basis:
                log.debug(2, "restart with the same basis.")
            else:
                log.debug(2, "restart with the different basis.")
            # umat maximally match the basis, C_old U ~ C_new
            basis_cas_old = basis_old.reshape(-1, nmo).dot(mo_coeff_old)
            basis_cas_new = basis_new.reshape(-1, nmo).dot(mo_coeff_new)
            umat = find_closest_mo(basis_cas_old, basis_cas_new, \
                    return_rotmat=True)[1]
        else: # UHF
            if bcc_restart:
                if is_same_basis:
                    mo_coeff_new = mo_coeff_old
                else:
                    basis_old = basis_old.reshape(spin, -1, nmo)
                    basis_new = basis_new.reshape(spin, -1, nmo)
                    umat = find_closest_mo(basis_old, basis_new, \
                            return_rotmat=True)[1]
                    mo_coeff_new = trans_mo(umat.conj().transpose(0, 2, 1), \
                            mo_coeff_old)
            
            if is_same_basis:
                log.debug(2, "restart with the same basis.")
            else:
                log.debug(2, "restart with the different basis.")
            basis_cas_old = trans_mo(basis_old.reshape(spin, -1, nmo),
                    mo_coeff_old)
            basis_cas_new = trans_mo(basis_new.reshape(spin, -1, nmo),
                    mo_coeff_new)
            umat = find_closest_mo(basis_cas_old, basis_cas_new, \
                    return_rotmat=True)[1]
        
        t1 = transform_t1_to_bo(t1_old, umat)
        t2 = transform_t2_to_bo(t2_old, umat)
        return t1, t2
    
    def save_t12_to_h5(self, fcc_name, basis_new, mo_coeff_new):
        """
        Save t1, t2, l1, l2, basis and mo_coeff.
        """
        log.debug(1, "CC solver: dump t and l")
        mo_coeff_new = np.asarray(mo_coeff_new)
        fcc = h5py.File(fcc_name, 'w')
        fcc['mo_coeff'] = mo_coeff_new
        fcc['basis'] = np.asarray(basis_new)
        if mo_coeff_new.ndim == 2: 
            fcc['t1'] = np.asarray(self.cisolver.t1)
            fcc['t2'] = np.asarray(self.cisolver.t2)
            fcc['l1'] = np.asarray(self.cisolver.l1)
            fcc['l2'] = np.asarray(self.cisolver.l2)
        else:
            spin = mo_coeff_new.shape[0]
            for s in range(spin):
                fcc['t1_%s'%s] = np.asarray(self.cisolver.t1[s])
                fcc['l1_%s'%s] = np.asarray(self.cisolver.l1[s])
            for s in range(spin*(spin+1)//2):
                fcc['t2_%s'%s] = np.asarray(self.cisolver.t2[s])
                fcc['l2_%s'%s] = np.asarray(self.cisolver.l2[s])
        fcc.close()

    def save_rdm_mo(self, rdm_fname='rdm_mo_cc.h5'):
        frdm = h5py.File(rdm_fname, 'w')
        frdm['rdm1'] = np.asarray(self.onepdm_mo)
        frdm['rdm2'] = np.asarray(self.twopdm_mo)
        frdm["mo_coeff"] = np.asarray(self.cisolver.mo_coeff)
        frdm.close()
    
    def load_rdm_mo(self, rdm_fname='rdm_mo_cc.h5'):
        frdm = h5py.File(rdm_fname, 'r')
        rdm1 = np.asarray(frdm["rdm1"])
        rdm2 = np.asarray(frdm["rdm2"])
        mo_coeff = np.asarray(frdm["mo_coeff"])
        frdm.close()
        return rdm1, rdm2, mo_coeff 
    
    def load_dmet_ham(self, dmet_ham_fname='dmet_ham.h5'):
        fdmet_ham = h5py.File(dmet_ham_fname, 'r')
        H1 = np.asarray(fdmet_ham["H1"])
        H2 = np.asarray(fdmet_ham["H2"])
        fdmet_ham.close()
        return H1, H2 
    
    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def cleanup(self):
        pass
        # FIXME first copy and save restart files

def get_umat_from_t1(t1):
    """
    Get rotation matrix, U = exp(t1 - t1^T)
    """
    if isinstance(t1, np.ndarray): # RHF GHF
        nocc, nvir = t1.shape
        amat = np.zeros((nocc+nvir, nocc+nvir), dtype=t1.dtype)
        amat[:nocc, -nvir:] = -t1
        amat[-nvir:, :nocc] = t1.conj().T
        umat = la.expm(amat)
    else: # UHF
        spin = len(t1)
        nmo = np.sum(t1[0].shape)
        umat = np.zeros((spin, nmo, nmo), dtype=np.result_type(*t1))
        for s in range(spin):
            nocc, nvir = t1[s].shape
            amat = np.zeros((nmo, nmo), dtype=t1[s].dtype)
            amat[:nocc, -nvir:] = -t1[s]
            amat[-nvir:, :nocc] = t1[s].conj().T
            umat[s] = la.expm(amat)
    return umat

def transform_t1_to_bo(t1, umat):
    """
    Transform t1 to brueckner orbital basis.
    """
    if isinstance(t1, np.ndarray) and t1.ndim == 2: # RHF GHF
        nocc, nvir = t1.shape
        umat_occ = umat[:nocc, :nocc]
        umat_vir = umat[nocc:, nocc:]   
        return mdot(umat_occ.conj().T, t1, umat_vir)
    else: # UHF
        spin = len(t1)
        return [transform_t1_to_bo(t1[s], umat[s]) \
                for s in range(spin)]

def transform_t2_to_bo(t2, umat, umat_b=None):
    """
    Transform t2 to brueckner orbital basis.
    4 dot, 7 reshape, 3 transpose
    """
    if isinstance(t2, np.ndarray) and t2.ndim == 4: # RHF GHF
        if umat_b is None:
            umat_b = umat
        umat_a = umat

        nocc_a, nocc_b, nvir_a, nvir_b = t2.shape
        umat_occ_a = umat_a[:nocc_a, :nocc_a]
        umat_occ_b = umat_b[:nocc_b, :nocc_b]
        umat_vir_a = umat_a[nocc_a:, nocc_a:]
        umat_vir_b = umat_b[nocc_b:, nocc_b:]
        
        tmp = np.empty((nocc_a*nocc_b*nvir_a, nvir_b), dtype=t2.dtype)
        # (ija)b, bB -> (ija)B
        lib.dot(t2.reshape(-1, nvir_b), umat_vir_b, c=tmp)
        # (ija)B) -> i(jaB)
        tmp = tmp.reshape(nocc_a, -1)
        # iI, i(jaB) -> I(jaB)
        np.dot(umat_occ_a.conj().T, tmp, out=tmp)
        # I(jaB) -> j(IaB)
        tmp = np.asarray(tmp.reshape(nocc_a, nocc_b, -1).transpose(1, 0, 2).\
                reshape(nocc_b, -1), order='C')
        # jJ, j(IaB) -> J(IaB)
        np.dot(umat_occ_b.T, tmp, out=tmp)
        # J(IaB) -> IJBa -> (IJB)a
        tmp = np.asarray(tmp.reshape(nocc_b, nocc_a, nvir_a, nvir_b).\
                transpose(1, 0, 3, 2).reshape(-1, nvir_a), order='C')
        # (IJB)a, aA -> (IJB)A
        np.dot(tmp, umat_vir_a.conj(), out=tmp)
        # (IJB)A -> IJBA -> IJAB
        t2_bo = tmp.reshape(nocc_a, nocc_b, nvir_b, nvir_a).transpose(0, 1, 3, 2)
    else: # UHF
        # t2 order: aa, ab, bb
        t2_bo = [None, None, None]
        t2_bo[0] = transform_t2_to_bo(t2[0], umat[0])
        t2_bo[1] = transform_t2_to_bo(t2[1], umat[0], umat_b=umat[1])
        t2_bo[2] = transform_t2_to_bo(t2[2], umat[1])
    return t2_bo

def bcc_loop(mycc, u=None, utol=1e-6, max_cycle=100, diis=True, verbose=2):
    def max_abs(x):
        if isinstance(x, np.ndarray):
            if np.iscomplexobj(x):
                return np.abs(x).max()
            else:
                return max(np.max(x), abs(np.min(x)))
        else:
            return np.max([max_abs(xi) for xi in x])

    if u is None:
        u = get_umat_from_t1(mycc.t1)
    mf = mycc._scf
    ovlp = mf.get_ovlp()
    adiis = lib.diis.DIIS()
    with lib.temporary_env(mf, verbose=verbose):
        e_tot_last = mycc.e_tot
        for i in range(max_cycle):
            mf.mo_coeff = trans_mo(mf.mo_coeff, u)
            if diis:
                mo_coeff_new = adiis.update(mf.mo_coeff)
                u = trans_mo(u, get_mo_ovlp(mf.mo_coeff, mo_coeff_new, ovlp))
                mf.mo_coeff = mo_coeff_new
            mf.e_tot = mf.energy_tot()
            t1 = transform_t1_to_bo(mycc.t1, u)
            t2 = transform_t2_to_bo(mycc.t2, u)
            mycc.__init__(mf)
            mycc.verbose = verbose
            mycc.kernel(t1=t1, t2=t2)
            dE = mycc.e_tot - e_tot_last
            e_tot_last = mycc.e_tot
            if not mycc.converged:
                log.warn("CC not converged")
            t1_norm = max_abs(mycc.t1)
            log.info("BCC iter: %4d  E: %20.12f  dE: %12.3e  |t1|: %12.3e", \
                    i, mycc.e_tot, dE, t1_norm)
            if t1_norm < utol:
                break
            u = get_umat_from_t1(mycc.t1)
        else:
            log.warn("BCC: not converged, max_cycle reached.")
    return mycc

if __name__ == '__main__':
    pass
