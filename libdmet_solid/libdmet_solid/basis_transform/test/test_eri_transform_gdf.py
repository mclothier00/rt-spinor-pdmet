#!/usr/bin/env python

'''
Test ERI transformation from k-space to real space
'''

import numpy as np
import scipy.linalg as la
import os, sys

from pyscf import lib
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df

from libdmet_solid.lo import iao, pywannier90
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.basis_transform import eri_transform
from libdmet_solid.utils.misc import mdot, max_abs
from libdmet_solid.utils import logger as log

np.set_printoptions(4, linewidth=1000, suppress=True)
log.verbose = "DEBUG2"

def _test_ERI(cell, gdf, kpts, C_ao_lo):
    # Fast ERI
    # Give a very small memory to force small blksize (for test only)
    eri_k2gamma = eri_transform.get_emb_eri(cell, gdf, C_ao_lo=C_ao_lo, \
            max_memory=0.02, t_reversal_symm=True)
    eri_k2gamma = eri_k2gamma[0]

    scell, phase = eri_transform.get_phase(cell, kpts)
    mydf_scell = df.GDF(scell)
    nao = cell.nao_nr()
    nkpts = len(kpts)
    smf = scf.KRHF(scell, exxdiv=None).density_fit()
    Cgamma = np.einsum('Rk, kui, Sk -> RuSi', phase, C_ao_lo, phase.conj())
    Cgamma = Cgamma.reshape(nao*nkpts, nao*nkpts)
    eri_scell = smf.with_df.ao2mo(Cgamma, compact=False).reshape([nao*nkpts]*4)
    print ('ERI Max Imag', max_abs(eri_scell.imag))
    print ('Fast ERI compared to supcell', max_abs(eri_scell-eri_k2gamma))
    assert max_abs(eri_scell.imag) < 1e-8
    assert max_abs(eri_scell - eri_k2gamma) < 1e-8

    ## Slow ERI
    #Lpq_kpts = eri_transform.get_Lpq_5d(cell, gdf)
    #Lpq_kpts = eri_transform.Lpq_5d_ao2mo(Lpq_kpts, C_ao_lo)
    #Lpq_6d = eri_transform.Lpq_5d_to_6d(cell, gdf.kpts, Lpq_kpts)
    #Lpq_sc = eri_transform.Lpq_6d_k2gamma_pq(cell, gdf.kpts, Lpq_6d)
    #eri_k2gamma = np.einsum('Lpq,Lrs->qprs', Lpq_sc.conj(), Lpq_sc)/nkpts
    #print 'Slow ERI compared to supcell', la.norm(eri_scell-eri_k2gamma)
    #assert la.norm(eri_scell-eri_k2gamma) < 1e-8

    # Fast 1st unit cell ERI (for DMFT)
    eri_k2gamma = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo, \
            t_reversal_symm=True)
    eri_k2gamma_no_tr = eri_transform.get_unit_eri(cell, gdf, \
            C_ao_lo=C_ao_lo, t_reversal_symm=False)
    eri_k2gamma = eri_k2gamma[0]
    print ('Fast 1st unit cell ERI compared to supcell',\
            max_abs(eri_scell[:nao,:nao,:nao,:nao] - eri_k2gamma))
    assert max_abs(eri_k2gamma - eri_k2gamma_no_tr) < 1e-10
    assert max_abs(eri_scell[:nao,:nao,:nao,:nao] - eri_k2gamma) < 1e-8

def test_ERI_gdf():
    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = 'ccpvdz'
    cell.verbose = 1
    cell.precision = 1e-14
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()
    nao = cell.nao_nr()

    print ('### Test AO ERI transform ###')
    C_ao_lo = np.zeros((nkpts, nao, nao),dtype=complex)
    C_ao_lo[:, range(nao), range(nao)] = 1.+0.j
    _test_ERI(cell, gdf, kpts, C_ao_lo)

#    print ('### Test Lowdin AO ERI transform ###')
#    ovlp = np.asarray(cell.pbc_intor('cint1e_ovlp_sph',kpts=kpts))
#    X = np.zeros_like(ovlp)
#    for ki in range(nkpts):
#        X[ki] = la.inv(la.sqrtm(ovlp[ki]))
#    C_ao_lo = X.copy()
#    _test_ERI(cell, gdf, kpts, C_ao_lo)

    print ('### Test IAO+PAO ERI transform ###')
    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    # IAO
    Lat = lattice.Lattice(cell, kmesh)
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, full_return=False)
    _test_ERI(cell, gdf, kpts, C_ao_lo)

    #print ('### Test Wannier ERI transform ###')
    #ncore = 0
    #nval = C_ao_lo_val.shape[-1]
    #nvirt = cell.nao_nr() - ncore - nval
    #Lat.set_val_virt_core(nval, nvirt, ncore)
    #C_ao_mo = np.asarray(kmf.mo_coeff)
    #
    ## use IAO orbital as Wannier's guess
    #A_val = pywannier90.get_A_mat_from_lo(C_ao_mo[:, :, Lat.val_idx], S, C_ao_lo_val)
    #A_virt = pywannier90.get_A_mat_from_lo(C_ao_mo[:, :, Lat.virt_idx], S, C_ao_lo_virt)
    #
    #proj_val = "H: l=0"
    ##proj_virt = "H: l=0;l=1"
    #proj_virt = "H: sp3"
    #C_ao_lo = make_basis.get_C_ao_lo_wannier(Lat, kmf, proj_val, proj_virt, num_iter=1000, A_val = A_val, A_virt=A_virt)
    #_test_ERI(cell, gdf, kpts, C_ao_lo)

if __name__ == "__main__":
    test_ERI_gdf()
