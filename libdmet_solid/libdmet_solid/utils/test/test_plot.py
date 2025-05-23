#! /usr/bin/env python

"""
Test plot.
"""

def test_plot_smooth():
    """
    Test plot_smooth.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from libdmet_solid.utils import plot_smooth

    x = [0.0, 0.5, 2.0, 4.0, 5.0, 7.0, 8.0]
    y = np.sin(x) + (np.random.random(len(x)) - 0.5) * 0.1
    
    x_plot, y_plot1 = plot_smooth(x, y, smooth=0.1, do_plot=False)
    x_plot, y_plot2 = plot_smooth(x, y, smooth=1e-2, n0left=20, remove_neg=True)
    plt.savefig("plot_smooth.png")
    x_plot, y_plot2 = plot_smooth(x, y, smooth=1e-2, n0right=30)

def test_dos():
    """
    Test pdos routine and plot.
    """
    import numpy as np
    import scipy.linalg as la
    from libdmet_solid.utils.plot import get_dos, plot_dos
    from libdmet_solid.utils.misc import max_abs
    
    # not use Xwindow as backend
    import matplotlib
    matplotlib.use('Agg')
    
    # fake MO and C_lo_mo
    mo_energy = np.asarray([[-2.0, -2.0, 0.0, 0.1, 0.5, 7.0], \
                            [-2.0, -0.1, 0.0, 0.1, 0.5, 6.5],
                            [-2.0, -0.1, 0.0, 0.1, 0.5, 6.5]])
    nkpts, nmo = mo_energy.shape
    C_lo_mo = np.random.random((nkpts, nmo, nmo))
    for k in range(nkpts):
        C_lo_mo[k] = la.qr(C_lo_mo[k])[0]
    idx_dic = {"%s"%(i): [i] for i in range(nmo)}
    color_dic = {"%s"%(i): 'C%s'%(i) for i in range(nmo)}
    color_dic_select = {"%s"%(i): 'C%s'%(i) for i in range(2)}
    elist, pdos = get_dos(mo_energy, ndos=201, mo_coeff=C_lo_mo, sigma=0.1)
    elist_spin, pdos_spin = get_dos(mo_energy[None], ndos=201, \
            mo_coeff=C_lo_mo[None], sigma=0.1)
    assert max_abs(elist - elist_spin) < 1e-10
    assert pdos_spin.ndim == pdos.ndim + 1 
    assert max_abs(elist - elist_spin) < 1e-10
    plot_dos(elist, pdos.sum(axis=0), idx_dic=None, text='test_total', fig_name='test.pdf')
    
    plot_dos(elist, pdos, idx_dic=idx_dic, text='test')
    
    plot_dos(elist, np.asarray((pdos, pdos * 0.7)), \
            idx_dic=idx_dic, color_dic=color_dic, fig_name='pdos_uhf.pdf')
    
    plot_dos(elist, np.asarray((pdos, pdos * 0.7)), \
            idx_dic=idx_dic, color_dic=None, \
            fig_name='pdos_uhf_no_color_dic.pdf')
    
    plot_dos(elist, np.asarray((pdos, pdos * 0.7)), \
            idx_dic=idx_dic, color_dic=color_dic_select, \
            fig_name='pdos_uhf_select_color_dic.pdf')

def test_cube():
    import os
    import numpy as np
    import scipy.linalg as la
    
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet_solid.system import lattice
    from libdmet_solid.basis_transform import make_basis
    from libdmet_solid.basis_transform import eri_transform

    from libdmet_solid.utils import logger as log
    import libdmet_solid.dmet.Hubbard as dmet
    from libdmet_solid.utils.misc import max_abs, mdot
    from libdmet_solid.utils.plot import plot_orb_k_all, plot_density_k

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = gto.Cell()
    cell.a = ''' 10.0     0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    exxdiv = None

    ### ************************************************************
    ### DMET settings 
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

    # DMET SCF control
    MaxIter = 1
    u_tol = 1.0e-6
    E_tol = 1.0e-6
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 100 # embedding fitting
    full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

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
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
        assert(kmf.converged)


    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO guess
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True, max_ovlp=True)

    # Wannier orbitals
    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    ### ************************************************************
    ### DMET procedure
    ### ************************************************************

    # DMET main loop
    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    if load_frecord:
        dmet.SolveImpHam_with_fitting.load("./frecord")

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
        Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath)

        # plot LOs or MOs
        latt_vec = cell.lattice_vectors()
        latt_vec[0,0] = 3.0
        latt_vec[1,1] = 3.0
        plot_orb_k_all(cell, 'iao_val', C_ao_iao_val, kpts, nx=50, ny=50, \
                nz=50, resolution=None, margin=5.0, latt_vec=latt_vec, \
                boxorig=[5.0, 5.0, -3.0])
        plot_orb_k_all(cell, 'iao_virt', C_ao_iao_virt, kpts, nx=50, ny=50, \
                nz=50, resolution=None, margin=5.0, latt_vec=latt_vec, \
                boxorig=[5.0, 5.0, -3.0])
        #plot_orb_k_all(cell, 'mo_val', C_ao_mo, kpts, nx=50, ny=50, nz=50, \
        #        resolution=None, margin=3.0,  latt_vec=latt_vec, \
        #        boxorig=[5.0, 5.0, -3.0])
        
        # plot rdm1_lo_R0
        scell = Lat.bigcell
        scell.verbose = 4
        basis_k = Lat.R2k_basis(basis)
        rdm1_lo_R0 = rho[0, 0] * 2.0
        C_ao_lo0 = Lat.k2R_basis(C_ao_lo)
        C_ao_lo0_full = C_ao_lo0.reshape((nkpts*nscsites, -1))
        rdm1_sc = mdot(C_ao_lo0_full, rdm1_lo_R0, C_ao_lo0_full.conj().T)
        plot_density_k(scell, 'rdm1_sc.cube', rdm1_sc[None], kpts_abs=[[0.0, 0.0, 0.0]], \
                nx=50, ny=50, nz=50, resolution=None, margin=3.0)
        
        # plot bath orbital density
        C_ao_emb_k = make_basis.multiply_basis(C_ao_lo[None], basis_k)
        C_ao_emb_R = Lat.k2R_basis(C_ao_emb_k).reshape((1, nkpts*nscsites, -1))
        C_bath = C_ao_emb_R[:, :, -nval:]
        dm_bath = (C_bath[0].dot(C_bath[0].conj().T))[None]
        
        plot_density_k(scell, 'rho.cube', dm_bath, kpts_abs=[[0.0, 0.0, 0.0]], \
                nx=50, ny=50, nz=50, resolution=None, margin=3.0)
        break

def test_plot_bands():
    """
    Plot band structure with PDOS of the 3-band Hubbard model.
    """
    import os
    import numpy as np
    import scipy.linalg as la

    import libdmet_solid.utils.logger as log
    import libdmet_solid.dmet.abinitioBCS as dmet
    from libdmet_solid.routine import mfd
    from libdmet_solid.utils import get_dos
    from libdmet_solid.utils import plot_bands

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    matplotlib.use('Agg')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    log.verbose = "DEBUG0"
    np.set_printoptions(3, linewidth=1000, suppress=True)

    doping = 0.0 
    Filling = (5.-doping) / 6
    Filling_Cu = (1.0 - doping) / 2.0

    LatSize = (10, 10)
    ImpSize = (1, 1)

    Lat = dmet.Square3BandSymm(*(LatSize + ImpSize))
    nao = Lat.nscsites
    nkpts = Lat.nkpts
    Ham = dmet.Hubbard3band_ref(Lat, "Hanke", min_model=False, ignore_intercell=False)
    Lat.setHam(Ham)
    vcor = dmet.vcor_zeros(False, True, Lat.supercell.nsites)

    # initial guess for HF
    nCu_tot = np.prod(LatSize) * 4 # 4 is number of Cu site per 2x2 cell
    nO_tot = np.prod(LatSize) * 8 
    nao_tot = nao * nkpts 
    nelec_half = np.prod(LatSize) * 20 # 20 electron per cell
    nelec_half_Cu = np.prod(LatSize) * 4 
    nelec_half_O = np.prod(LatSize) * 16

    x_dop = 0.0
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

    polar = 0.5
    fCu_a = Filling_Cu * (1.0 - polar)
    fCu_b = Filling_Cu * (1.0 + polar)
    fO = Filling_O

    dm0_a = np.diag([fCu_a, fO, fO, fCu_b, fO, fO, fCu_a, fO, fO, fCu_b, fO, fO])
    dm0_b = np.diag([fCu_b, fO, fO, fCu_a, fO, fO, fCu_b, fO, fO, fCu_a, fO, fO])
    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    # HF calcs
    if os.path.exists("./dm0.npy"):
        dm0 = np.load("dm0.npy")
    rho, mu, E, res = mfd.HF(Lat, vcor, Filling, False, mu0=0.0, \
            beta=np.inf, ires = True, scf=True, dm0=dm0)
    np.save("dm0.npy", res["rho_k"])
    rdm1_a, rdm1_b = rho[:, 0]
    m_AFM = 0.25 * (abs(rdm1_a[0, 0] - rdm1_b[0, 0]) + abs(rdm1_a[3, 3] - rdm1_b[3, 3]) \
                   +abs(rdm1_a[6, 6] - rdm1_b[6, 6]) + abs(rdm1_a[9, 9] - rdm1_b[9, 9]))
    log.result("m_AFM = %s", m_AFM)


    # *****************************************
    # Plot bands and pdos
    # *****************************************

    mo_occ = (np.abs(res["mo_occ"] - 1.0) < 1e-6)
    gap = res["gap"]
    ew = res["e"]
    ev = res["coef"]
    # shift by VBM
    vbm = ew[mo_occ].max()
    ew -= vbm

    # K-path:
    # G (0, 0) -> X (0.5, 0.0) -> S(0.5, 0.5) -> G(0, 0)
    # GX, XS, SG
    kpts = Lat.kpts_scaled
    kpts[kpts == -0.5] = 0.5

    GX_idx = np.where((kpts[:, 0] >= 0.0)  & (kpts[:, 1] == 0))[0][:-1]
    XS_idx = np.where((kpts[:, 0] == 0.5)  & (kpts[:, 1] >= 0))[0][:-1]
    SG_idx = np.where((kpts[:, 0] >=  0.0) & (kpts[:, 1] == kpts[:, 0]))[0][::-1]

    log.result("G-X indices:\n%s", GX_idx)
    log.result("G-X kpts:\n%s", kpts[GX_idx])
    log.result("X-S indices:\n%s", XS_idx)
    log.result("X-S kpts:\n%s", kpts[XS_idx])
    log.result("S-G indices:\n%s", SG_idx)
    log.result("S-G kpts:\n%s", kpts[SG_idx])

    kpath_idx = np.hstack((GX_idx, XS_idx, SG_idx))
    kpts_bands = kpts[kpath_idx]
    kdis = np.diff(kpts_bands, axis=0)
    kdis = np.hstack((0.0, np.cumsum(la.norm(kdis, axis=1))))
    G_idx = np.where((kpts_bands[:, 0] == 0.0) & (kpts_bands[:, 1] == 0))[0][0]
    X_idx = np.where((kpts_bands[:, 0] == 0.5) & (kpts_bands[:, 1] == 0))[0][0]
    S_idx = np.where((kpts_bands[:, 0] == 0.5) & (kpts_bands[:, 1] == 0.5))[0][0]
    sp_points_idx = np.hstack((G_idx, X_idx, S_idx, -1))
    sp_points = kdis[sp_points_idx]

    sigma = 0.01
    ew_bands = (ew[0, kpath_idx])
    ev_bands = ev[0, kpath_idx]

    ev_bands_O_sq = la.norm(ev_bands[:, [1, 2, 4, 5, 7, 8, 10, 11]], axis=1) ** 2
    ev_bands_Cu_sq = la.norm(ev_bands[:, [0, 3, 6, 9]], axis=1) ** 2
    ev_bands_Cu_percent = ev_bands_Cu_sq #/ (ev_bands_Cu_sq + ev_bands_O_sq)
    ev_bands_O_percent = ev_bands_O_sq #/ (ev_bands_Cu_sq + ev_bands_O_sq)

    mo_energy_min = ew.min()
    mo_energy_max = ew.max()
    margin = max(10 * sigma, 0.05 * (mo_energy_max - mo_energy_min)) # margin
    emin = -8.5
    emax = 4.5
    nbands = ew_bands.shape[-1]

    fig, ax = plt.subplots(figsize=(6, 5), sharey=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2])
    ax = plt.gca()

    """
    Color mapped bands:
    """
    plt.subplot(gs[0])
    plt.tick_params(labelsize=20, bottom=False, top=False, left=True, right=True, width=1.5)
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    bands = plot_bands(ax, kdis, ew_bands, weights=ev_bands_Cu_percent, \
            cmap='coolwarm', linewidth=4)

    for p in sp_points[1:3]:
        # special lines grids
        plt.plot([p, p], [emin, emax], '--', color='lightgrey', linewidth=2.0, zorder=-10)

    plt.xticks(sp_points, [r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
    plt.axis(xmin=0, xmax=sp_points[-1], ymin=-8.5, ymax=4.5)
    plt.yticks(np.arange(-8, 6, 2))
    plt.xlabel("$\mathbf{k}$", fontsize=20)
    plt.ylabel("energy [eV]", fontsize=20)

    plt.text(0.03, 0.95, 'HF', horizontalalignment='left', \
            verticalalignment='center', transform=ax.transAxes, fontsize=20)

    """
    PDOS:
    """

    Es, pdos = get_dos(ew, ndos=3001, sigma=0.05, mo_coeff=ev)
    pdos = pdos.sum(axis=0)
    np.save("elist-hf.npy", Es)
    np.save("pdos-hf.npy", pdos)

    DOS_Cu = pdos[[0, 3, 6, 9], :]
    DOS_O = pdos[[1, 2, 4, 5, 7, 8, 10, 11], :]
    DOS_Cu = np.sum(DOS_Cu, axis=0)
    DOS_O = np.sum(DOS_O, axis=0)

    plt.subplot(gs[1])
    ax = plt.gca()
    ax.set_facecolor('white') # background color

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tick_params(labelsize=20, bottom='off', top='off', left=True, right=True, width=1.5)
    plt.xlabel("PDOS", fontsize=20, labelpad=7)
    plt.xticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.xlim(0.0, 7.5)
    plt.ylim(emin, emax)
    plt.yticks(np.arange(-8, 6, 2))

    cmap = matplotlib.cm.get_cmap('coolwarm')
    rgba1 = cmap(0.95)
    rgba2 = cmap(0.05)

    plt.plot(DOS_Cu, Es, label='Cu $3d$', linewidth=2, color=rgba1, linestyle='dashed')
    plt.plot(DOS_O,  Es, label='O   $2p$', linewidth=2, color=rgba2)

    """
    Color bar:
    """
    cbar = fig.colorbar(bands, ax=ax, aspect=35, pad=0.1, ticks=[0.025, 0.975])
    cbar.outline.set_linewidth(1.5)
    cbar.ax.set_yticklabels(['O', 'Cu'])
    cbar.ax.tick_params(labelsize=20, bottom=True, top=True, left=False,
            right=False, width=1.5)

    plt.subplots_adjust(left=0.15, bottom=0.165, right=0.94, top=0.97, wspace=0.1, hspace=0.0)
    #plt.show()
    plt.tight_layout()
    plt.savefig("./bands-dos-HF.eps", dpi=300)
    
if __name__ == "__main__":
    test_plot_bands()
    test_plot_smooth()
    test_dos()
    test_cube()
