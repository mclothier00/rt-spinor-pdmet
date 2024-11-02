#!/usr/bin/env python

import os, sys
import numpy as np
import scipy.linalg as la

np.set_printoptions(4, linewidth=1000)

def test_neighbor():
    from libdmet_solid.system.lattice import ChainLattice, SquareLattice, ham
    from libdmet_solid.utils import logger as log
    
    chain = ChainLattice(240, 4)
    log.result("%s", chain)
    log.result("kpoints: %s", chain.kpoints())
    log.result("neighbors: %s", chain.neighbor(sitesA = range(4)))
    log.result("")

    square = SquareLattice(72, 72, 2, 2)
    log.result("%s", square)
    log.result("1st neigbors: %s", square.neighbor(dis = 1., sitesA = range(4)))
    log.result("2nd neigbors: %s", square.neighbor(dis = 2**0.5, sitesA = range(4)))

    Ham = ham.HubbardHamiltonian(square, 4, [1, 0.])
    square.setHam(Ham)
    log.result("%s", square.getH1(kspace = True))
    log.result("%s", square.getFock(kspace = True))
    log.result("%s", square.getH2())

def test_k2R():
    from pyscf.pbc import gto as gto
    import pyscf.pbc.scf as scf
    import pyscf.pbc.df as df
    from pyscf.pbc.lib import chkfile

    from libdmet_solid.basis_transform import make_basis
    from libdmet_solid.system import lattice
    from libdmet_solid.utils.misc import max_abs
    from libdmet_solid.routine.mfd import get_eri_7d, get_j_from_eri_7d, \
            get_k_from_eri_7d
    
    cell = lattice.HChain()
    cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts = Lat.kpts_abs
    nkpts = Lat.nkpts
    exxdiv = None

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()
    
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=False)
    
    # check the lo
    Lat.check_lo(C_ao_lo, ovlp=kmf.get_ovlp())
    C_ao_lo_symm_first_real = Lat.symmetrize_lo(C_ao_lo, real_first=True)
    C_ao_lo_symm_after_real = Lat.symmetrize_lo(C_ao_lo, real_first=False)
    assert max_abs(C_ao_lo_symm_first_real - C_ao_lo) < 1e-10
    assert max_abs(C_ao_lo_symm_after_real - C_ao_lo) < 1e-10
    
    C_ao_lo_R = Lat.k2R(C_ao_lo)

    mo_coeff = np.asarray(kmf.mo_coeff)
    hcore_k = kmf.get_hcore()
    dm_k = kmf.make_rdm1()
    nao = cell.nao_nr()
    nmo = mo_coeff.shape[-1]

    mo_id = np.zeros_like(mo_coeff)
    mo_id[:, range(nmo), range(nmo)] = 1.0
    eri_7d_ref = gdf.ao2mo_7d(mo_id, kpts=kpts)
    vj_ref, vk_ref = gdf.get_jk(dm_k) 
    
    eri_7d = get_eri_7d(cell, gdf)
    vj = get_j_from_eri_7d(eri_7d, dm_k)
    vk = get_k_from_eri_7d(eri_7d, dm_k)
    
    diff_eri = max_abs(eri_7d - eri_7d_ref)
    diff_vj = max_abs(vj - vj_ref)
    diff_vk = max_abs(vk - vk_ref)
    print ("eri 7d diff (from ref): ", diff_eri)
    print ("vj diff: ", diff_vj)
    print ("vk diff: ", diff_vk)
    assert(diff_eri < 1e-10)
    assert(diff_vj < 1e-10)
    assert(diff_vk < 1e-10)
    
    hcore_R = Lat.k2R(hcore_k)
    hcore_R_T = Lat.transpose(hcore_R)
    hcore_R_T = Lat.transpose(hcore_R[None])
    hcore_R_full = Lat.expand(hcore_R)
    dm_R = Lat.k2R(dm_k)
    dm_R_full = Lat.expand(dm_R)

    # supercell calculation
    scell = Lat.bigcell
    kmesh_sc = [1, 1, 1]  
    Lat_sc = lattice.Lattice(scell, kmesh_sc)

    mf = scf.KRHF(scell, exxdiv=exxdiv)
    mf = mf.density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    
    hcore_R_ref = mf.get_hcore()
    dm_R_ref = mf.make_rdm1()
    C_ao_lo_sc = make_basis.get_C_ao_lo_iao(Lat_sc, mf, minao='minao', full_return=False)
    C_ao_lo_R_ref = Lat.extract_stripe(C_ao_lo_sc)

    diff_e = abs(kmf.e_tot - mf.e_tot/nkpts)
    print ("max diff energy: %s" %diff_e)
    assert diff_e < 1e-5
    
    diff_dm = max_abs(dm_R_full - dm_R_ref)
    print ("max diff between kpts and gamma rdm: %s" %diff_dm)
    assert diff_dm < 1e-5
    
    diff_hcore = max_abs(hcore_R_full - hcore_R_ref)
    print ("max diff between kpts and gamma hcore: %s" %diff_hcore)
    assert diff_hcore < 1e-5 
    
    diff_C_ao_lo = max_abs(C_ao_lo_R - C_ao_lo_R_ref)
    print ("max diff between C_ao_lo_R and C_ao_lo_R_ref(sc): %s" %diff_C_ao_lo)
    assert diff_C_ao_lo < 1e-5

def test_k2R_H2():
    """
    Test k2R and R2k for two-particle objects.
    """
    import pyscf.pbc.df as df
    from libdmet_solid.system import lattice
    from libdmet_solid.utils.misc import max_abs
    from libdmet_solid.routine.mfd import get_eri_7d
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    cell = lattice.HChain()
    cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nao = nscsites = Lat.nscsites
    kpts = Lat.kpts_abs
    nkpts = Lat.nkpts
    exxdiv = None

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
    #if True:
        gdf.build()
    
    eri_7d_k = get_eri_7d(cell, gdf)
    eri_7d_R = Lat.k2R_H2(eri_7d_k)
    
    # supercell reference
    gdf_sc_fname = 'gdf_sc_ints.h5'
    gdf_sc = df.GDF(Lat.bigcell, np.zeros((1, 3)))
    gdf_sc._cderi_to_save = gdf_sc_fname
    if not os.path.isfile(gdf_sc_fname):
    #if True:
        gdf_sc.build()
    
    # ERI R
    eri_sc = gdf_sc.get_eri(compact=False)\
            .reshape(nkpts, nao, nkpts, nao, nkpts, nao, nkpts, nao)
    eri_sc_7d = eri_sc[:, :, :, :, :, :, 0].transpose(0, 2, 4, 1, 3, 5, 6)
    print (max_abs(eri_7d_R - eri_sc_7d))
    assert max_abs(eri_7d_R - eri_sc_7d) < 1e-12

    # ERI k
    eri_7d_k_re = Lat.R2k_H2(eri_7d_R)
    print (max_abs(eri_7d_k_re - eri_7d_k))
    assert max_abs(eri_7d_k_re - eri_7d_k) < 1e-12

def test_analyze():
    """
    Test analyze.
    """
    from pyscf.pbc import gto as gto
    import pyscf.pbc.scf as scf
    import pyscf.pbc.df as df
    from pyscf.pbc.lib import chkfile

    from libdmet_solid.lo import iao
    from libdmet_solid.basis_transform import make_basis
    from libdmet_solid.system import lattice
    from libdmet_solid.utils.misc import max_abs
    from libdmet_solid.system.lattice import analyze_kmo

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''

    cell.atom = ''' H 0.      0.      0.
                    H 0.      0.      1.5 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')
   
    # lattice class
    kmesh = [1, 1, 3]  
    Lat = lattice.Lattice(cell, kmesh)
    nscsites = Lat.nscsites
    kpts = Lat.kpts_abs
    nkpts = Lat.nkpts
    exxdiv = None

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.chkfile = chkfname
        kmf.kernel()
    
    ovlp = np.asarray(kmf.get_ovlp())
    # test RHF with meta Lowdin
    print ("\nTest meta-lowdin RHF analysis")
    pop_rhf, chg_rhf = Lat.analyze(kmf)
    
    # test RHF with IAOs
    print ("\nTest IAO+PAO RHF analysis")
    minao = 'minao'
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao)
    labels = iao.get_labels(cell, minao)[0]
    idx_to_ao_labels = iao.get_idx_to_ao_labels(cell, minao)
    Lat.analyze(kmf, C_ao_lo=C_ao_lo, labels=labels)
    
    # analyze each MO
    analyze_kmo(kmf, C_ao_lo, labels)
    idx_dic = iao.get_idx_each(cell, minao=minao, kind='atom nl')
    analyze_kmo(kmf, C_ao_lo, idx_dic, k_loop_first=False)
    Lat.analyze_kmo(kmf, num_max=1, k_loop_first=False)

    rdm1_ao = np.asarray(kmf.make_rdm1())
    rdm1_ao_R0 = Lat.k2R(rdm1_ao)[0]
    rdm1_lo_R0 = Lat.k2R(make_basis.transform_rdm1_to_lo(rdm1_ao, C_ao_lo, \
            ovlp))[0]
    print ("\nTest IAO+PAO RHF compare density")
    Lat.mulliken_lo_R0(rdm1_lo_R0=rdm1_ao_R0, rdm1_lo_R0_2=rdm1_lo_R0, \
            labels=labels)
    
    # test UHF with meta Lowdin
    print ("\nTest IAO+PAO UHF analysis")
    kmf = scf.addons.convert_to_uhf(kmf) 
    pop_uhf, chg_uhf = Lat.analyze(kmf)
    assert max_abs(pop_uhf[0] + pop_uhf[1] - pop_rhf) < 1e-10
    assert max_abs(chg_uhf - chg_rhf) < 1e-10
    
    # test UHF with IAOs, 1 valence and 1 virtual
    print ("\nTest IAO+PAO UHF analysis 1 valence 1 virtual")
    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao)
    idx = [0, 3]
    labels = [labels[i] for i in idx]
    Lat.analyze(kmf, C_ao_lo=C_ao_lo[:, :, :, idx], labels=labels)

    print ("\nTest IAO+PAO UHF compare density")
    labels = iao.get_labels(cell, minao)[0]
    rdm1_lo_R0 = np.asarray((rdm1_lo_R0, rdm1_lo_R0)) * 0.5
    Lat.mulliken_lo_R0(rdm1_lo_R0=rdm1_lo_R0, rdm1_lo_R0_2=rdm1_lo_R0, \
            labels=labels)
    Lat.mulliken_lo_R0(rdm1_lo_R0=rdm1_lo_R0, rdm1_lo_R0_2=rdm1_lo_R0, \
            labels=labels)

def test_diff_kmesh_odd():
    """
    Test different kmesh, odd kmesh.
    """
    from pyscf.pbc import gto, scf, df
    from pyscf.pbc.lib import chkfile
    from libdmet_solid.system import lattice
    from libdmet_solid.system.lattice import get_R_vec, \
            find_idx_k_in_K, find_idx_R_vec, k2gamma, fold_kmf, fold_h1, \
            fold_lo
    from libdmet_solid.basis_transform import make_basis, eri_transform
    from libdmet_solid.utils import logger as log
    from libdmet_solid.utils.misc import max_abs
     
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"
    
    basis = '321G'

    cell_0 = gto.Cell()
    cell_0.a    = '''   5.0    0.0    0.0
                        0.0    5.0    0.0
                        0.0    0.0    3.0 '''
    cell_0.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5 '''
    cell_0.basis = basis
    cell_0.verbose = 4
    cell_0.precision = 1e-10
    cell_0.build(unit='Angstrom')
    
    kmesh_0 = [2, 1, 6]
    Lat_0 = lattice.Lattice(cell_0, kmesh_0)
    kpts_0 = Lat_0.kpts
    nao_0 = Lat_0.nao
    nkpts_0 = Lat_0.nkpts
    
    cell_1 = gto.Cell()
    cell_1.a    = '''  10.0    0.0    0.0
                        0.0    5.0    0.0
                        0.0    0.0    6.0 '''
    # be careful with the order of atom arrangement.
    cell_1.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5
                      H 0.0    0.0    3.0
                      H 0.0    0.0    4.5 
                      H 5.0    0.0    0.0
                      H 5.0    0.0    1.5 
                      H 5.0    0.0    3.0
                      H 5.0    0.0    4.5 '''
    cell_1.basis = basis
    cell_1.verbose = 4
    cell_1.precision = 1e-10
    cell_1.build(unit='Angstrom')

    kmesh_1 = [1, 1, 3]
    Lat_1 = lattice.Lattice(cell_1, kmesh_1)
    kpts_1 = Lat_1.kpts
    nao_1 = Lat_1.nao
    nkpts_1 = Lat_1.nkpts
    
    # Convert the mf objects from small cell @ large kmesh to 
    # supercell @ small kmesh
    exxdiv = None
    gdf_fname_0 = 'gdf_ints_0.h5'
    gdf_0 = df.GDF(cell_0, kpts_0)
    gdf_0._cderi_to_save = gdf_fname_0
    #if not os.path.isfile(gdf_fname_0):
    if True:
        gdf_0.build()
    chkfname_0 = 'hchain_0.chk'
    kmf_0 = scf.KRHF(cell_0, kpts_0, exxdiv=exxdiv)
    kmf_0.with_df = gdf_0
    kmf_0.with_df._cderi = gdf_fname_0
    kmf_0.conv_tol = 1e-12
    kmf_0.chkfile = chkfname_0
    #if not os.path.isfile(chkfname_0):
    if True:
        kmf_0.kernel()
    else:
        data = chkfile.load(chkfname_0, 'scf')
        kmf_0.__dict__.update(data)
    ew_0 = kmf_0.mo_energy
    ev_0 = kmf_0.mo_coeff
    occ_0 = kmf_0.mo_occ

    # Then check smaller kmesh
    print ("\nReference small kmesh calculation")
    ew_1, ev_1, occ_1 = fold_kmf(ew_0, ev_0, occ_0, Lat_0, Lat_1, tol=1e-10)
    
    gdf_fname_1 = 'gdf_ints_1.h5'
    gdf_1 = df.GDF(cell_1, kpts_1)
    gdf_1._cderi_to_save = gdf_fname_1
    #if not os.path.isfile(gdf_fname_1):
    if True:
        gdf_1.build()
    chkfname_1 = 'hchain_1.chk'
    kmf_1 = scf.KRHF(cell_1, kpts_1, exxdiv=exxdiv)
    kmf_1.with_df = gdf_1
    kmf_1.with_df._cderi = gdf_fname_1
    kmf_1.conv_tol = 1e-11
    kmf_1.chkfile = chkfname_1
    dm0 = kmf_1.make_rdm1(ev_1, occ_1)
    ovlp_1 = kmf_1.get_ovlp()
    nelec_1 = np.einsum('kij, kji ->', dm0, ovlp_1) / Lat_1.nkpts
    assert (nelec_1 - cell_1.nelectron) < 1e-10
    kmf_1.max_cycle = 1 # should be already converged
    #if not os.path.isfile(chkfname_1):
    if True:
        kmf_1.kernel(dm0)
    else:
        data = chkfile.load(chkfname_1, 'scf')
        kmf_1.__dict__.update(data)
    
    diff_energy = (kmf_1.e_tot * nkpts_1) - (kmf_0.e_tot * nkpts_0)
    print ("Energy diff: ", diff_energy)
    assert diff_energy < 1e-9
    
    # check fold_h1
    hcore_0 = kmf_0.get_hcore()
    hcore_1 = fold_h1(hcore_0, Lat_0, Lat_1)
    hcore_1_ref = kmf_1.get_hcore()
    diff_hcore = max_abs(hcore_1 - hcore_1_ref)
    print ("hcore difference: ", diff_hcore)
    assert diff_hcore < 1e-9
    
    # check fold_rdm1
    rdm1_0 = kmf_0.make_rdm1()
    rdm1_1 = fold_h1(rdm1_0, Lat_0, Lat_1)
    rdm1_1_ref = kmf_1.make_rdm1()
    diff_rdm1 = max_abs(rdm1_1 - rdm1_1_ref)
    print ("rdm1 difference: ", diff_rdm1)
    assert diff_rdm1 < 1e-7

    # check fold C_ao_lo
    C_ao_lo_0, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_0,\
            kmf_0, minao='minao', full_return=True)
    C_ao_lo_R_0 = Lat_0.k2R(C_ao_lo_0)
    ncore_0 = 0
    nval_0 = C_ao_iao_val.shape[-1]
    nvirt_0 = cell_0.nao_nr() - ncore_0 - nval_0
    Lat_0.set_val_virt_core(nval_0, nvirt_0, ncore_0)
    
    C_ao_lo_1, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_1,\
            kmf_1, minao='minao', full_return=True)
    C_ao_lo_R_1 = Lat_1.k2R(C_ao_lo_1)
    ncore_1 = 0
    nval_1 = C_ao_iao_val.shape[-1]
    nvirt_1 = cell_1.nao_nr() - ncore_1 - nval_1
    Lat_1.set_val_virt_core(nval_1, nvirt_1, ncore_1)
    
    C_ao_lo_1_fold = fold_lo(C_ao_lo_0, Lat_0, Lat_1)
    diff_C_ao_lo = max_abs(C_ao_lo_1_fold - C_ao_lo_1)
    print ("C_ao_lo difference: ", diff_C_ao_lo)
    assert diff_C_ao_lo < 1e-7
    
    C_ao_lo_sc_ref = Lat_0.expand_orb(Lat_0.k2R(C_ao_lo_0))
    C_ao_lo_sc_fold = Lat_1.expand_orb(Lat_1.k2R(C_ao_lo_1))
    assert max_abs(np.sort(C_ao_lo_sc_ref, kind='mergesort') - \
            np.sort(C_ao_lo_sc_fold, kind='mergesort')) < 1e-7

    # integral transform
    C_ao_LO = fold_lo(C_ao_lo_0, Lat_0, Lat_1, uc2sc=True)
    hcore_LO_R = Lat_1.k2R(make_basis.transform_h1_to_lo(hcore_1_ref, \
            C_ao_lo_1))
    
    hcore_LO_k_fold = make_basis.transform_h1_to_lo(hcore_0, C_ao_LO)
    hcore_LO_R0_fold = hcore_LO_k_fold.sum(axis=0) / nkpts_0
    diff_hcore_R0 = max_abs(hcore_LO_R0_fold - hcore_LO_R[0])
    print ("hcore_LO_R0 diff: ", diff_hcore_R0)
    assert diff_hcore_R0 < 1e-8
    
    eri_1_ref = eri_transform.get_unit_eri_fast(cell_1, gdf_1, C_ao_lo_1,
            symmetry=1, t_reversal_symm=True)
    eri_1 = eri_transform.get_unit_eri_fast(cell_0, gdf_0, C_ao_LO,
            symmetry=1, t_reversal_symm=True)
    diff_int = max_abs(eri_1 - eri_1_ref)
    print ("diff integral transform: ", diff_int)
    assert diff_int < 1e-8

def test_diff_kmesh_even():
    """
    Test different kmesh, even kmesh.
    """
    from pyscf.pbc import gto, scf, df
    from libdmet_solid.system import lattice
    from libdmet_solid.system.lattice import get_R_vec, \
            find_idx_k_in_K, find_idx_R_vec, k2gamma, fold_kmf, fold_h1, \
            fold_lo
    from libdmet_solid.basis_transform import make_basis, eri_transform
    from libdmet_solid.utils import logger as log
    from libdmet_solid.utils.misc import max_abs
     
    np.set_printoptions(4, linewidth=1000, suppress=False)
    log.verbose = "DEBUG2"
    
    basis = 'sto3g'

    cell_0 = gto.Cell()
    cell_0.a    = '''   3.0    0.0    0.0
                        0.0    3.0    0.0
                        0.0    0.0    3.0 '''
    cell_0.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5 '''
    cell_0.basis = basis
    cell_0.verbose = 4
    cell_0.precision = 1e-10
    cell_0.build(unit='Angstrom')
    
    kmesh_0 = [1, 1, 4]
    Lat_0 = lattice.Lattice(cell_0, kmesh_0)
    kpts_0 = Lat_0.kpts
    nao_0 = Lat_0.nao
    nkpts_0 = Lat_0.nkpts
    
    cell_1 = gto.Cell()
    cell_1.a    = '''   3.0    0.0    0.0
                        0.0    3.0    0.0
                        0.0    0.0    6.0 '''
    cell_1.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5
                      H 0.0    0.0    3.0
                      H 0.0    0.0    4.5 '''
    cell_1.basis = basis
    cell_1.verbose = 4
    cell_1.precision = 1e-10
    cell_1.build(unit='Angstrom')

    kmesh_1 = [1, 1, 2]
    Lat_1 = lattice.Lattice(cell_1, kmesh_1)
    kpts_1 = Lat_1.kpts
    nao_1 = Lat_1.nao
    nkpts_1 = Lat_1.nkpts
    
    cell_2 = gto.Cell()
    cell_2.a    = '''   3.0    0.0    0.0
                        0.0    3.0    0.0
                        0.0    0.0    12.0 '''
    cell_2.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5
                      H 0.0    0.0    3.0
                      H 0.0    0.0    4.5 
                      H 0.0    0.0    6.0
                      H 0.0    0.0    7.5
                      H 0.0    0.0    9.0 
                      H 0.0    0.0    10.5 '''
    cell_2.basis = basis
    cell_2.verbose = 4
    cell_2.precision = 1e-10
    cell_2.build(unit='Angstrom')

    kmesh_2 = [1, 1, 1]
    Lat_2 = lattice.Lattice(cell_2, kmesh_2)
    kpts_2 = Lat_2.kpts
    nao_2 = Lat_2.nao
    nkpts_2 = Lat_2.nkpts
    
    idx_k_in_1 = find_idx_k_in_K(kpts_0, kpts_1, cell_1)
    print ("idx k for cell 1")
    print (idx_k_in_1)
    
    R_vec_0 = get_R_vec(cell_0, kmesh_0)
    idx_R_in_1 = find_idx_R_vec(R_vec_0, kmesh_1, cell_1)
    print ("idx R for cell 1")
    print (idx_R_in_1)
    
    idx_k_in_2 = find_idx_k_in_K(kpts_0, kpts_2, cell_2)
    print ("idx k for cell 2")
    print (idx_k_in_2)
    assert (idx_k_in_2 == 0).all()

    R_vec_0 = get_R_vec(cell_0, kmesh_0)
    idx_R_in_2 = find_idx_R_vec(R_vec_0, kmesh_2, cell_2)
    print ("idx R for cell 2")
    print (idx_R_in_2)
    assert (idx_R_in_2 == 0).all()

    # Convert the mf objects from small cell @ large kmesh to 
    # supercell @ small kmesh
    exxdiv = None
    gdf_fname_0 = 'gdf_ints_0.h5'
    gdf_0 = df.GDF(cell_0, kpts_0)
    gdf_0._cderi_to_save = gdf_fname_0
    gdf_0.build()
    chkfname_0 = 'hchain.chk'
    kmf_0 = scf.KRHF(cell_0, kpts_0, exxdiv=exxdiv)
    kmf_0.with_df = gdf_0
    kmf_0.with_df._cderi = gdf_fname_0
    kmf_0.conv_tol = 1e-10
    kmf_0.chkfile = chkfname_0
    kmf_0.kernel()
    ew_0 = kmf_0.mo_energy
    ev_0 = np.asarray(kmf_0.mo_coeff)
    occ_0 = kmf_0.mo_occ

    ## First check Gamma point, k2gamma
    print ("\nReference Gamma point calculation")
    ew_g, ev_g, occ_g = k2gamma(ew_0, ev_0, occ_0, Lat_0.phase, \
            make_real=True, lattice=Lat_0, ovlp=kmf_0.get_ovlp())
#    ew_2, ev_2, occ_2 = fold_kmf(ew_0, ev_0, occ_0, Lat_0, Lat_2, tol=1e-10)
#    
#    gdf_fname_2 = 'gdf_ints_2.h5'
#    gdf_2 = df.GDF(cell_2, kpts_2)
#    gdf_2._cderi_to_save = gdf_fname_2
#    gdf_2.build()
#    chkfname_2 = 'hchain.chk'
#    kmf_2 = scf.KRHF(cell_2, kpts_2, exxdiv=exxdiv)
#    kmf_2.with_df = gdf_2
#    kmf_2.with_df._cderi = gdf_fname_2
#    kmf_2.conv_tol = 1e-10
#    kmf_2.chkfile = chkfname_2
#    dm0 = kmf_2.make_rdm1(ev_g, occ_g)
#    dm0_fold = kmf_2.make_rdm1(ev_2, occ_2)
#    assert max_abs(dm0 - dm0_fold) < 1e-10
#    kmf_2.max_cycle = 1 # should be already converged
#    kmf_2.kernel(dm0)
#    
#    diff_energy = (kmf_2.e_tot * nkpts_2) - (kmf_0.e_tot * nkpts_0)
#    print ("Energy diff: ", diff_energy)
#    assert diff_energy < 1e-9
    
    # Then check smaller kmesh
    print ("\nReference small kmesh calculation")
    ew_1, ev_1, occ_1 = fold_kmf(ew_0, ev_0, occ_0, Lat_0, Lat_1, tol=1e-10)
    
    gdf_fname_1 = 'gdf_ints_1.h5'
    gdf_1 = df.GDF(cell_1, kpts_1)
    gdf_1._cderi_to_save = gdf_fname_1
    gdf_1.build()
    chkfname_1 = 'hchain.chk'
    kmf_1 = scf.KRHF(cell_1, kpts_1, exxdiv=exxdiv)
    kmf_1.with_df = gdf_1
    kmf_1.with_df._cderi = gdf_fname_1
    kmf_1.conv_tol = 1e-10
    kmf_1.chkfile = chkfname_1
    dm0 = kmf_1.make_rdm1(ev_1, occ_1)
    ovlp_1 = kmf_1.get_ovlp()
    nelec_1 = np.einsum('kij, kji ->', dm0, ovlp_1) / Lat_1.nkpts
    assert (nelec_1 - cell_1.nelectron) < 1e-10
    kmf_1.max_cycle = 1 # should be already converged
    kmf_1.kernel(dm0)
    
    diff_energy = (kmf_1.e_tot * nkpts_1) - (kmf_0.e_tot * nkpts_0)
    print ("Energy diff: ", diff_energy)
    assert diff_energy < 1e-9
    
    # check fold_h1
    hcore_0 = kmf_0.get_hcore()
    hcore_1 = fold_h1(hcore_0, Lat_0, Lat_1)
    hcore_1_ref = kmf_1.get_hcore()
    diff_hcore = max_abs(hcore_1 - hcore_1_ref)
    print ("hcore difference: ", diff_hcore)
    assert diff_hcore < 1e-9
    
    # check fold_rdm1
    rdm1_0 = kmf_0.make_rdm1()
    rdm1_1 = fold_h1(rdm1_0, Lat_0, Lat_1)
    rdm1_1_ref = kmf_1.make_rdm1()
    diff_rdm1 = max_abs(rdm1_1 - rdm1_1_ref)
    print ("rdm1 difference: ", diff_rdm1)
    assert diff_rdm1 < 1e-7
    
    # check fold C_ao_lo
    C_ao_lo_0, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_0,\
            kmf_0, minao='minao', full_return=True)
    C_ao_lo_R_0 = Lat_0.k2R(C_ao_lo_0)
    ncore_0 = 0
    nval_0 = C_ao_iao_val.shape[-1]
    nvirt_0 = cell_0.nao_nr() - ncore_0 - nval_0
    Lat_0.set_val_virt_core(nval_0, nvirt_0, ncore_0)
    
    C_ao_lo_1, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_1,\
            kmf_1, minao='minao', full_return=True)
    C_ao_lo_R_1 = Lat_1.k2R(C_ao_lo_1)
    ncore_1 = 0
    nval_1 = C_ao_iao_val.shape[-1]
    nvirt_1 = cell_1.nao_nr() - ncore_1 - nval_1
    Lat_1.set_val_virt_core(nval_1, nvirt_1, ncore_1)
    
    C_ao_lo_1_fold = fold_lo(C_ao_lo_0, Lat_0, Lat_1)
    diff_C_ao_lo = (max_abs(C_ao_lo_1_fold - C_ao_lo_1))
    print ("C_ao_lo difference: ", diff_C_ao_lo)
    assert diff_C_ao_lo < 1e-7
    
    ## check fold C_ao_mo 
    ## this will be different since MOs' phase is undetermined.
    #mo_coeff_0 = np.asarray(kmf_0.mo_coeff)
    #mo_coeff_1 = fold_h1(mo_coeff_0, Lat_0, Lat_1)
    #mo_coeff_1_ref = np.asarray(kmf_1.mo_coeff)
    #diff_mo_coeff = max_abs(mo_coeff_1 - mo_coeff_1_ref)
    #print ("mo_coeff difference: ", diff_mo_coeff)
    
    # integral transform
    C_ao_LO = fold_lo(C_ao_lo_0, Lat_0, Lat_1, uc2sc=True)
    hcore_LO_R = Lat_1.k2R(make_basis.transform_h1_to_lo(hcore_1_ref, \
            C_ao_lo_1))
    
    hcore_LO_k_fold = make_basis.transform_h1_to_lo(hcore_0, C_ao_LO)
    hcore_LO_R0_fold = hcore_LO_k_fold.sum(axis=0) / nkpts_0
    diff_hcore_R0 = max_abs(hcore_LO_R0_fold - hcore_LO_R[0])
    print ("hcore_LO_R0 diff: ", diff_hcore_R0)
    assert diff_hcore_R0 < 1e-8
    
    eri_1_ref = eri_transform.get_unit_eri_fast(cell_1, gdf_1, C_ao_lo_1,
            symmetry=1, t_reversal_symm=True)
    eri_1 = eri_transform.get_unit_eri_fast(cell_0, gdf_0, C_ao_LO,
            symmetry=1, t_reversal_symm=True)
    diff_int = max_abs(eri_1 - eri_1_ref)
    print ("diff integral transform: ", diff_int)
    assert diff_int < 1e-8

def test_diff_kmesh_integral():
    """
    Test different kmesh, integral transform.
    """
    from pyscf.pbc import gto, scf, df
    from libdmet_solid.system import lattice
    from libdmet_solid.system.lattice import get_R_vec, \
            find_idx_k_in_K, find_idx_R_vec, k2gamma, fold_kmf, fold_h1, \
            fold_lo
    from libdmet_solid.basis_transform import make_basis, eri_transform
    from libdmet_solid.utils import logger as log
    from libdmet_solid.utils.misc import max_abs
     
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"
    
    basis = '321g'

    cell_0 = gto.Cell()
    cell_0.a    = '''   5.0    0.0    0.0
                        0.0    5.0    0.0
                        0.0    0.0    3.0 '''
    cell_0.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5 '''
    cell_0.basis = basis
    cell_0.verbose = 4
    cell_0.precision = 1e-10
    cell_0.build(unit='Angstrom')
    
    kmesh_0 = [1, 1, 6]
    Lat_0 = lattice.Lattice(cell_0, kmesh_0)
    kpts_0 = Lat_0.kpts
    nao_0 = Lat_0.nao
    nkpts_0 = Lat_0.nkpts
    
    cell_1 = gto.Cell()
    cell_1.a    = '''   5.0    0.0    0.0
                        0.0    5.0    0.0
                        0.0    0.0    9.0 '''
    cell_1.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5
                      H 0.0    0.0    3.0
                      H 0.0    0.0    4.5
                      H 0.0    0.0    6.0
                      H 0.0    0.0    7.5 '''
    cell_1.basis = basis
    cell_1.verbose = 4
    cell_1.precision = 1e-10
    cell_1.build(unit='Angstrom')

    kmesh_1 = [1, 1, 2]
    Lat_1 = lattice.Lattice(cell_1, kmesh_1)
    kpts_1 = Lat_1.kpts
    nao_1 = Lat_1.nao
    nkpts_1 = Lat_1.nkpts
    
    # Convert the mf objects from small cell @ large kmesh to 
    # supercell @ small kmesh
    exxdiv = None
    gdf_fname_0 = 'gdf_ints_0.h5'
    gdf_0 = df.GDF(cell_0, kpts_0)
    gdf_0._cderi_to_save = gdf_fname_0
    #if not os.path.isfile(gdf_fname_0):
    if True:
        gdf_0.build()
    chkfname_0 = 'hchain.chk'
    kmf_0 = scf.KRHF(cell_0, kpts_0, exxdiv=exxdiv)
    kmf_0.with_df = gdf_0
    kmf_0.with_df._cderi = gdf_fname_0
    kmf_0.conv_tol = 1e-10
    kmf_0.chkfile = chkfname_0
    kmf_0.kernel()
    ew_0 = kmf_0.mo_energy
    ev_0 = np.asarray(kmf_0.mo_coeff)
    occ_0 = kmf_0.mo_occ
    
    # Then check smaller kmesh
    print ("\nReference small kmesh calculation")
    ew_1, ev_1, occ_1 = fold_kmf(ew_0, ev_0, occ_0, Lat_0, Lat_1, tol=1e-10)
    
    gdf_fname_1 = 'gdf_ints_1.h5'
    gdf_1 = df.GDF(cell_1, kpts_1)
    gdf_1._cderi_to_save = gdf_fname_1
    #if not os.path.isfile(gdf_fname_1):
    if True:
        gdf_1.build()
    chkfname_1 = 'hchain.chk'
    kmf_1 = scf.KRHF(cell_1, kpts_1, exxdiv=exxdiv)
    kmf_1.with_df = gdf_1
    kmf_1.with_df._cderi = gdf_fname_1
    kmf_1.conv_tol = 1e-10
    kmf_1.chkfile = chkfname_1
    dm0 = kmf_1.make_rdm1(ev_1, occ_1)
    kmf_1.max_cycle = 1 # should be already converged
    kmf_1.kernel(dm0)
    
    diff_energy = (kmf_1.e_tot * nkpts_1) - (kmf_0.e_tot * nkpts_0)
    print ("Energy diff: ", diff_energy)
    assert diff_energy < 1e-9
    
    # check fold_h1
    hcore_0 = kmf_0.get_hcore()
    hcore_1 = fold_h1(hcore_0, Lat_0, Lat_1)
    hcore_1_ref = kmf_1.get_hcore()
    diff_hcore = max_abs(hcore_1 - hcore_1_ref)
    print ("hcore difference: ", diff_hcore)
    assert diff_hcore < 1e-9
    
    # check fold_rdm1
    rdm1_0 = kmf_0.make_rdm1()
    rdm1_1 = fold_h1(rdm1_0, Lat_0, Lat_1)
    rdm1_1_ref = kmf_1.make_rdm1()
    diff_rdm1 = max_abs(rdm1_1 - rdm1_1_ref)
    print ("rdm1 difference: ", diff_rdm1)
    assert diff_rdm1 < 1e-7
    
    # check fold C_ao_lo
    C_ao_lo_0, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_0,\
            kmf_0, minao='minao', full_return=True)
    ncore_0 = 0
    nval_0 = C_ao_iao_val.shape[-1]
    nvirt_0 = cell_0.nao_nr() - ncore_0 - nval_0
    Lat_0.set_val_virt_core(nval_0, nvirt_0, ncore_0)
    
    C_ao_lo_1, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_1,\
            kmf_1, minao='minao', full_return=True)
    ncore_1 = 0
    nval_1 = C_ao_iao_val.shape[-1]
    nvirt_1 = cell_1.nao_nr() - ncore_1 - nval_1
    Lat_1.set_val_virt_core(nval_1, nvirt_1, ncore_1)
    
    # integral transform
    eri_1_ref = eri_transform.get_unit_eri_fast(cell_1, gdf_1, C_ao_lo_1,
            symmetry=1)
    
    C_ao_LO = fold_lo(C_ao_lo_0, Lat_0, Lat_1, uc2sc=True)
    eri_1 = eri_transform.get_unit_eri_fast(cell_0, gdf_0, C_ao_LO,
            symmetry=1)
    
    diff_int = max_abs(eri_1 - eri_1_ref)
    print ("diff integral transform: ", diff_int)
    assert diff_int < 1e-9

def test_diff_kmesh_integral_2():
    """
    Test different kmesh, odd kmesh.
    """
    from pyscf.pbc import gto, scf, df
    from pyscf.pbc.lib import chkfile
    from libdmet_solid.system import lattice
    from libdmet_solid.system.lattice import get_R_vec, \
            find_idx_k_in_K, find_idx_R_vec, k2gamma, fold_kmf, fold_h1, \
            fold_lo
    from libdmet_solid.basis_transform import make_basis, eri_transform
    from libdmet_solid.utils import logger as log
    from libdmet_solid.utils.misc import max_abs, mdot
     
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"
    
    basis = 'minao'

    cell_0 = gto.Cell()
    cell_0.a    = '''   5.0    0.0    0.0
                        0.0    5.0    0.0
                        0.0    0.0    3.0 '''
    cell_0.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5 '''
    cell_0.basis = basis
    cell_0.verbose = 4
    cell_0.precision = 1e-10
    cell_0.build(unit='Angstrom')
    
    kmesh_0 = [1, 2, 4]
    Lat_0 = lattice.Lattice(cell_0, kmesh_0)
    kpts_0 = Lat_0.kpts
    nao_0 = Lat_0.nao
    nkpts_0 = Lat_0.nkpts
    
    cell_1 = gto.Cell()
    cell_1.a    = '''   5.0    0.0    0.0
                        0.0    10.0    0.0
                        0.0    0.0    6.0 '''
    # be careful with the order of atom arrangement.
    cell_1.atom = ''' H 0.0    0.0    0.0
                      H 0.0    0.0    1.5
                      H 0.0    0.0    3.0
                      H 0.0    0.0    4.5
                      H 0.0    5.0    0.0
                      H 0.0    5.0    1.5
                      H 0.0    5.0    3.0
                      H 0.0    5.0    4.5'''
    cell_1.basis = basis
    cell_1.verbose = 4
    cell_1.precision = 1e-10
    cell_1.build(unit='Angstrom')

    kmesh_1 = [1, 1, 2]
    Lat_1 = lattice.Lattice(cell_1, kmesh_1)
    kpts_1 = Lat_1.kpts
    nao_1 = Lat_1.nao
    nkpts_1 = Lat_1.nkpts
    
    # Convert the mf objects from small cell @ large kmesh to 
    # supercell @ small kmesh
    exxdiv = None
    gdf_fname_0 = 'gdf_ints_0.h5'
    gdf_0 = df.GDF(cell_0, kpts_0)
    gdf_0._cderi_to_save = gdf_fname_0
    #if not os.path.isfile(gdf_fname_0):
    if True:
        gdf_0.build()
    chkfname_0 = 'hchain_0.chk'
    kmf_0 = scf.KRHF(cell_0, kpts_0, exxdiv=exxdiv)
    kmf_0.with_df = gdf_0
    kmf_0.with_df._cderi = gdf_fname_0
    kmf_0.conv_tol = 1e-12
    kmf_0.chkfile = chkfname_0
    #if not os.path.isfile(chkfname_0):
    if True:
        kmf_0.kernel()
    else:
        data = chkfile.load(chkfname_0, 'scf')
        kmf_0.__dict__.update(data)

    ew_0 = kmf_0.mo_energy
    ev_0 = kmf_0.mo_coeff
    occ_0 = kmf_0.mo_occ

    # Then check smaller kmesh
    print ("\nReference small kmesh calculation")
    ew_1, ev_1, occ_1 = fold_kmf(ew_0, ev_0, occ_0, Lat_0, Lat_1, tol=1e-10)
    
    gdf_fname_1 = 'gdf_ints_1.h5'
    gdf_1 = df.GDF(cell_1, kpts_1)
    gdf_1._cderi_to_save = gdf_fname_1
    #if not os.path.isfile(gdf_fname_1):
    if True:
        gdf_1.build()
    chkfname_1 = 'hchain_1.chk'
    kmf_1 = scf.KRHF(cell_1, kpts_1, exxdiv=exxdiv)
    kmf_1.with_df = gdf_1
    kmf_1.with_df._cderi = gdf_fname_1
    kmf_1.conv_tol = 1e-11
    kmf_1.chkfile = chkfname_1
    dm0 = kmf_1.make_rdm1(ev_1, occ_1)
    ovlp_1 = kmf_1.get_ovlp()
    nelec_1 = np.einsum('kij, kji ->', dm0, ovlp_1) / Lat_1.nkpts
    assert (nelec_1 - cell_1.nelectron) < 1e-10
    kmf_1.max_cycle = 1 # should be already converged
    #if not os.path.isfile(chkfname_1):
    if True:
        kmf_1.kernel(dm0)
    else:
        data = chkfile.load(chkfname_1, 'scf')
        kmf_1.__dict__.update(data)
    
    diff_energy = (kmf_1.e_tot * nkpts_1) - (kmf_0.e_tot * nkpts_0)
    print ("Energy diff: ", diff_energy)
    assert diff_energy < 1e-9
    
    # check fold_h1
    hcore_0 = kmf_0.get_hcore()
    hcore_1 = fold_h1(hcore_0, Lat_0, Lat_1)
    hcore_1_ref = kmf_1.get_hcore()
    diff_hcore = max_abs(hcore_1 - hcore_1_ref)
    print ("hcore difference: ", diff_hcore)
    assert diff_hcore < 1e-9
    
    # check fold_rdm1
    rdm1_0 = kmf_0.make_rdm1()
    rdm1_1 = fold_h1(rdm1_0, Lat_0, Lat_1)
    rdm1_1_ref = kmf_1.make_rdm1()
    diff_rdm1 = max_abs(rdm1_1 - rdm1_1_ref)
    print ("rdm1 difference: ", diff_rdm1)
    assert diff_rdm1 < 1e-6

    # check fold C_ao_lo
    C_ao_lo_0, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_0,\
            kmf_0, minao='minao', full_return=True)
    C_ao_lo_R_0 = Lat_0.k2R(C_ao_lo_0)
    ncore_0 = 0
    nval_0 = C_ao_iao_val.shape[-1]
    nvirt_0 = cell_0.nao_nr() - ncore_0 - nval_0
    Lat_0.set_val_virt_core(nval_0, nvirt_0, ncore_0)
    
    C_ao_lo_1, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat_1,\
            kmf_1, minao='minao', full_return=True)
    C_ao_lo_R_1 = Lat_1.k2R(C_ao_lo_1)
    ncore_1 = 0
    nval_1 = C_ao_iao_val.shape[-1]
    nvirt_1 = cell_1.nao_nr() - ncore_1 - nval_1
    Lat_1.set_val_virt_core(nval_1, nvirt_1, ncore_1)
    
    C_ao_lo_1_fold = fold_lo(C_ao_lo_0, Lat_0, Lat_1)
    diff_C_ao_lo = max_abs(C_ao_lo_1_fold - C_ao_lo_1)
    print ("C_ao_lo difference: ", diff_C_ao_lo)
    assert diff_C_ao_lo < 1e-7
    
    # integral transform
    C_ao_LO = fold_lo(C_ao_lo_0, Lat_0, Lat_1, uc2sc=True)
    hcore_LO_R = Lat_1.k2R(make_basis.transform_h1_to_lo(hcore_1_ref, \
            C_ao_lo_1))
    
    hcore_LO_k_fold = make_basis.transform_h1_to_lo(hcore_0, C_ao_LO)
    hcore_LO_R0_fold = hcore_LO_k_fold.sum(axis=0) / nkpts_0
    diff_hcore_R0 = max_abs(hcore_LO_R0_fold - hcore_LO_R[0])
    print ("hcore_LO_R0 diff: ", diff_hcore_R0)
    assert diff_hcore_R0 < 1e-8
    
    eri_1_ref = eri_transform.get_unit_eri_fast(cell_1, gdf_1, C_ao_lo_1,
            symmetry=1, t_reversal_symm=True)
    eri_1 = eri_transform.get_unit_eri_fast(cell_0, gdf_0, C_ao_LO,
            symmetry=1, t_reversal_symm=True)
    diff_int = max_abs(eri_1 - eri_1_ref)
    print ("diff integral transform: ", diff_int)
    assert diff_int < 1e-8

def test_symm_orb():
    import numpy as np
    import scipy.linalg as la

    from pyscf import gto, lib

    from libdmet_solid.utils import logger as log
    from libdmet_solid.utils.misc import max_abs, mdot
    from libdmet_solid.system import lattice
    from libdmet_solid.system import hamiltonian as ham
    from libdmet_solid.dmet import Hubbard as dmet
    
    log.verbose = "DEBUG1"

    x_dop  = 0.0
    beta = 1000.0

    Ud   = 12.0
    Up   = 5.25
    ed   = -5.25
    tpd  = -1.5
    tpp  = -0.75
    tpp1 = 0.0
    Vpd  = 0.75

    # Lattice settings
    LatSize = [5, 5]
    ImpSize = [1, 1]

    Lat = dmet.Square3BandSymm(*(LatSize+ImpSize))
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize) * 12

    # Hamiltonian
    nCu_tot = np.prod(LatSize) * 4 # 4 is number of Cu site per 2x2 cell
    nO_tot = np.prod(LatSize) * 8
    nao_tot = nao * nkpts 
    nelec_half = np.prod(LatSize) * 20 # 20 electron per cell
    nelec_half_Cu = np.prod(LatSize) * 4
    nelec_half_O = np.prod(LatSize) * 16

    nelec_dop = int(np.round(x_dop * nCu_tot))
    if nelec_dop % 2 == 1:
        diff_l = abs(nelec_dop - 1 - x_dop * nCu_tot)
        diff_r = abs(nelec_dop + 1 - x_dop * nCu_tot)
        if diff_l < diff_r:
            nelec_dop = nelec_dop - 1
        else:
            nelec_dop = nelec_dop + 1
    x_dop = nelec_dop / float(nCu_tot)

    Filling = (nelec_half - nelec_dop) / (nao_tot * 2.0)
    if nelec_dop >= 0: # hole doping
        Filling_Cu = (nelec_half_Cu) / (nCu_tot * 2.0)
        Filling_O = (nelec_half_O - nelec_dop) / (nO_tot * 2.0)
    else: # electron doping
        Filling_Cu = (nelec_half_Cu - nelec_dop) / (nCu_tot * 2.0)
        Filling_O = (nelec_half_O) / (nO_tot * 2.0)

    log.info("doping x = %s", x_dop)
    log.info("nelec_half = %s", nelec_half)
    log.info("nelec_dop = %s", nelec_dop)

    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    Lat.cell.verbose = 4

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)
    vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites)
    
    polar = 0.5
    # make AFM guess dm0
    fCu_a = Filling_Cu * (1.0 - polar)
    fCu_b = Filling_Cu * (1.0 + polar)
    fO = Filling_O

    dm0_a = np.diag([fCu_a, fO, fO, fCu_b, fO, fO, fCu_a, fO, fO, fCu_b, fO, fO])
    dm0_b = np.diag([fCu_b, fO, fO, fCu_a, fO, fO, fCu_b, fO, fO, fCu_a, fO, fO])

    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=beta, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-8)
    log.result("\n2D Hubbard E per site (KUHF): %s \n", (Lat.kmf_lo.e_zero / natm))

    rdm1_a, rdm1_b = rho[:, 0]
    m_AFM = 0.25 * (abs(rdm1_a[0, 0] - rdm1_b[0, 0]) + abs(rdm1_a[3, 3] - rdm1_b[3, 3]) \
                   +abs(rdm1_a[6, 6] - rdm1_b[6, 6]) + abs(rdm1_a[9, 9] - rdm1_b[9, 9]))

    log.result("m_AFM = %s", m_AFM)

    veff = Lat.kmf_lo.get_veff()
    veff_R = Lat.k2R(veff) 
    #                  0      1      2      3
    vD = np.array([[ 0.0 , -0.01,  0.0,   0.01],
                   [-0.01,  0.0 ,  0.01 , 0.0 ],
                   [ 0.0,   0.01 , 0.0 , -0.01],
                   [ 0.01 , 0.0,  -0.01,  0.0 ]])


    vcor_mat = np.zeros((2, nao, nao))

    frac = 1.0 / 3.0
    vcor_mat[0] = (veff_R[0, 0] * (1.0 - frac) + veff_R[1, 0] * frac)
    vcor_mat[1] = (veff_R[0, 0] * frac + veff_R[1, 0] * (1.0 - frac))
    vcor.assign(vcor_mat)
    vcor_a, vcor_b = vcor.get()

    mol = gto.Mole()
    mol.atom = """
          Cu1 1.000000     1.00000      0.0000
          O1  0.000000     1.00000      0.0000
          O2  1.000000     2.00000      0.0000
          Cu2 1.000000     3.00000      0.0000
          O1  1.000000     4.00000      0.0000
          O2  2.000000     3.00000      0.0000
          Cu1 3.000000     3.00000      0.0000
          O1  4.000000     3.00000      0.0000
          O2  3.000000     2.00000      0.0000
          Cu2 3.000000     1.00000      0.0000
          O1  3.000000     0.00000      0.0000
          O2  2.000000     1.00000      0.0000
          """
    mol.symmetry = True
    mol.basis = {"Cu1": "def2-svp@1d", "Cu2": "def2-svp@1d", \
                  "O1": "def2-svp@1p", "O2": "def2-svp@1p"}
    mol.build()

    Cu1_3d = mol.search_ao_label("Cu1 3dx2-y2")
    Cu2_3d = mol.search_ao_label("Cu2 3dx2-y2")

    px_idx = []
    for i in [1, 5, 7, 11]:
        px_idx.append(mol.search_ao_label("%d O. 2px"%i)[0])

    py_idx = []
    for i in [2, 4, 8, 10]:
        py_idx.append(mol.search_ao_label("%d O. 2py"%i)[0])

    tband_idx   = np.hstack((Cu1_3d[0], px_idx[0], py_idx[0], \
                             Cu2_3d[0], py_idx[1], px_idx[1], \
                             Cu1_3d[1], px_idx[2], py_idx[2], \
                             Cu2_3d[1], py_idx[3], px_idx[3]))
    
    symm_orb_tband = lattice.get_symm_orb(mol, tband_idx)

    vcor_irep = []
    vcor_irep_param = []
    for i in range(len(symm_orb_tband)):
        C = symm_orb_tband[i]
        vcor_irep.append(mdot(C.conj().T, vcor_a, C))
        tril_idx = np.tril_indices(vcor_irep[-1].shape[-1])
        vcor_irep_param.append(vcor_irep[-1][tril_idx])
        vcor_irep.append(mdot(C.conj().T, vcor_b, C))
        tril_idx = np.tril_indices(vcor_irep[-1].shape[-1])
        vcor_irep_param.append(vcor_irep[-1][tril_idx])

    vcor_irep_param = np.hstack(vcor_irep_param)
    
    vcor = dmet.VcorSymm(restricted, bogoliubov, nscsites, symm_orb_tband, idx_range=None, \
            bogo_res=False)
    vcor.update(vcor_irep_param)
    vcor_re = vcor.get()
    
    print ("diff:", max_abs(vcor_re - vcor_mat))
    print (vcor.diag_indices())
    assert (max_abs(vcor_re - vcor_mat) < 1e-10)

def test_search_basis_id_sc():
    from pyscf.pbc import gto as gto
    from libdmet_solid.system import lattice
    from libdmet_solid.utils.misc import max_abs

    cell = gto.Cell()
    cell.a = ''' 2.0     -2.0     0.0
                 2.0     2.0      0.0
                 0.0     0.0     10.0 '''

    cell.atom = '''
          Cu 1.0  0.0  5. 
          Cu 3.0  0.0  5.
          O  2.0 -2.0  5.
          O  2.0  0.0  5.
          O  1.0  1.0  5.
          O  3.0  1.0  5.
          '''
    cell.basis = 'minao'
    cell.verbose = 4
    cell.build(unit='Angstrom')

    kmesh = [2, 2, 1]
    Lat = lattice.Lattice(cell, kmesh)
    scell = Lat.bigcell
    
    coords_target = np.array([
       [0.25,    0.25,    0.50], 
       [0.75,    0.75,    0.50], 
       [0.50,    0.50,    0.50], 
       [0.00,    0.50,    0.50], 
       [0.50,    1.00,    0.50], 
       [0.00,    0.00,    0.50],
       [1.00,    1.00,    0.50],
       [0.50,    0.00,    0.50],
       [1.00,    0.50,    0.50]])
    
    basis_idx = lattice.search_basis_id_sc(cell, scell, coords_target, \
            cell.ao_labels(), tol=1e-8)
    print (basis_idx)

if __name__ == '__main__':
    test_search_basis_id_sc()
    test_k2R_H2()
    test_symm_orb()
    test_analyze()
    test_k2R()
    test_neighbor()
    test_diff_kmesh_odd()
    test_diff_kmesh_even()
    test_diff_kmesh_integral()
    test_diff_kmesh_integral_2()

