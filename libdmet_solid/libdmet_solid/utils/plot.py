#! /usr/bin/env python

"""
Vistualization code.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from pyscf import lib
from pyscf.pbc.dft import numint
from libdmet_solid.utils import cubegen
from libdmet_solid.utils.lattice_plot import LatticePlot, plot_3band_order
from matplotlib import pyplot as plt

# ****************************************************************************
# plot curve
# ****************************************************************************

def plot_smooth(x, y, x_plot=None, label=None, color='black', marker='o', \
        linestyle='-', smooth=1e-4, remove_neg=False, n0left=None, \
        n0right=None, do_plot=True, **kwargs):
    """
    Plot a y-x curve with spline.
    
    Args:
        x: x 
        y: y
        x_plot: a list of fine mesh of x, if not provide, will be a linspace of 
                x with 100 points.
        label:  [None]
        color:  ['black']
        marker: ['o']
        linestyle: ['-']
        smooth: s for spline.
        remove_neg: remove the negative values to 0.
        n0left: left zero points.
        n0right: right zero points.
        do_plot: plot.
    
    Returns:
        x_plot: dense x points.
        y_plot: spline curve.
    """
    spl = UnivariateSpline(x, y, s=smooth)
    if x_plot is None:
        x_plot = np.linspace(np.min(x), np.max(x), 100)
    y_plot = spl(x_plot)
    if remove_neg:
        y_plot[y_plot < 0.0] = 0.0
    if n0left is not None:
        y_plot[:n0left] = 0.0
    if n0right is not None:
        y_plot[-n0right:] = 0.0
    
    if do_plot:
        plt.plot(x_plot, y_plot, color=color, marker='', linestyle=linestyle, **kwargs)
        plt.plot(x, y, color=color, marker=marker, linestyle='', label=label, **kwargs)
    return x_plot, y_plot

# ****************************************************************************
# plot periodic orbitals
# ****************************************************************************

def plot_orb_k(cell, outfile, coeff, kpts_abs, nx=80, ny=80, nz=80, resolution=None, margin=5.0, \
        latt_vec=None, boxorig=None, box=None):
    """
    Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """
    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin, \
            latt_vec=latt_vec, boxorig=boxorig, box=box)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = np.empty(ngrids)
    coeff /= np.sqrt(len(kpts_abs))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        #ao = numint.eval_ao(mol, coords[ip0:ip1])
        #orb_on_grid[ip0:ip1] = np.dot(ao, coeff)
        ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
        if ao.ndim == 2: # 1 kpt
            ao = ao[np.newaxis]
        orb_on_grid[ip0:ip1] = np.tensordot(ao, coeff, axes=((0,2), (0,1))).real

    orb_on_grid = orb_on_grid.reshape(cc.nx, cc.ny, cc.nz) 
    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')

def plot_orb_k_all(cell, outfile, coeffs, kpts_abs, nx=80, ny=80, nz=80, resolution=None, margin=5.0,\
        latt_vec=None, boxorig=None, box=None):
    """
    Plot all k-dependent orbitals in the reference cell.
    """
    coeffs = np.asarray(coeffs)
    nmo = coeffs.shape[-1]
    if len(coeffs.shape) == 3:
        for i in range(nmo):
            plot_orb_k(cell, outfile+"_mo%s.cube"%i, coeffs[:,:,i], kpts_abs, \
                    nx=nx, ny=ny, nz=nz, resolution=resolution, margin=margin, \
                    latt_vec=latt_vec, boxorig=boxorig, box=box)
    else:
        spin = coeff.shape[0]
        for s in range(spin):
            for i in range(nmo):
                plot_orb_k(cell, outfile+"_spin%s_mo%s.cube"%(s, i), coeffs[s,:,:,i], kpts_abs, \
                        nx=nx, ny=ny, nz=nz, resolution=resolution, margin=margin, \
                        latt_vec=latt_vec, boxorig=boxorig, box=box)

def plot_density_k(cell, outfile, dm, kpts_abs, nx=80, ny=80, nz=80, resolution=None, margin=5.0,\
        latt_vec=None, boxorig=None, box=None, skip_calc=False):
    """
    Calculates electron density and write out in cube format.

    Args:
        cell : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """

    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin, \
            latt_vec=latt_vec, boxorig=boxorig, box=box)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)
    if skip_calc:
        rho[:] = 0.0
    else:
        kni = numint.KNumInt(kpts_abs)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
            if ao.ndim == 2: # 1 kpt
                ao = ao[np.newaxis]
            #rho[ip0:ip1] = numint.eval_rho(cell, ao, dm)
            rho[ip0:ip1] = kni.eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=1)
    
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')

def get_ao_g_mol(mol, nx=40, ny=40, nz=40, resolution=None, coords=None):
    cc = cubegen.Cube(mol, nx, ny, nz, resolution)
    if coords is None:
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
    else:
        ngrids = coords.shape[0]
    blksize = min(8000, ngrids)
    nao = mol.nao_nr()
    orb_on_grid = np.empty((ngrids, nao))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = ao
    return orb_on_grid

def get_ao_g_k(cell, outfile, dm, kpts_abs, nx=80, ny=80, nz=80, resolution=None, margin=5.0,\
        latt_vec=None, boxorig=None, box=None, coords=None):
    cc = cubegen.Cube(cell, nx, ny, nz, resolution, margin=margin, \
            latt_vec=latt_vec, boxorig=boxorig, box=box)

    # Compute density on the .cube grid
    if coords is None:
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
    else:
        ngrids = coords.shape[0]

    blksize = min(8000, ngrids)
    nao = mol.nao_nr()
    nkpts = len(kpts_abs)
    orb_on_grid = np.empty((nkpts, ngrids, nao))
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = np.asarray(cell.pbc_eval_gto('GTOval', coords[ip0:ip1], kpts=kpts_abs))
        if ao.ndim == 2: # 1 kpt
            ao = ao[np.newaxis]
        orb_on_grid[:, ip0:ip1] = ao
    return orb_on_grid

# ****************************************************************************
# plot spin-spin correlation functions
# ****************************************************************************

def get_rho_pair(ao_g, mo):
    assert (mo.ndim == 3)
    ngrids = ao_g.shape[0]
    spin, nao, nmo = mo.shape
    mo_g = np.zeros((spin, ngrids, nmo))
    for s in range(spin):
        mo_g[s] = ao_g.dot(mo[s])
    rho_sji = np.einsum('sgi, sgj -> sgji', mo_g.conj(), mo_g)
    return rho_sji 

def eval_spin_corr_func_R(rho_pair0, rho_pair, rdm1, rdm2):
    """
    Spin-spin correlation function, <S(r0) S(r)>. Origin is r0.
    """
    spin, ng2, nmo, _ = rho_pair.shape
    cf = np.zeros((ng2,), dtype=rho_pair.dtype)
    
    # one-body terms
    rho_pair0 = rho_pair0.reshape((spin, nmo, nmo))
    rg0 = np.zeros((spin, nmo, nmo))
    for s in range(spin):
        rg0[s] = rho_pair0[s].conj().T.dot(rdm1[s])
        #cf += np.trace(rho_pair[s].dot(rg0[s]), axis1=1, axis2=2)
        cf += lib.einsum('gij, ji -> g', rho_pair[s], rg0[s])
    
    for s in range(spin):
        #tr2 = np.trace(rho_pair[s].dot(rdm1[s]), axis1=1, axis2=2)
        tr2 = lib.einsum('gij, ji -> g', rho_pair[s], rdm1[s])
        for t in range(spin):
            factor = (-1)**s * (-1)**t
            tr1 = rg0[t].trace()
            cf -= (factor*tr1)*tr2

    # two-body terms:
    rdm2_red = np.zeros((spin*spin, nmo, nmo))
    # t=0, s=0, ss=0
    rdm2_red[0] = lib.einsum('sr, srpq -> pq', rho_pair0[0].conj(), rdm2[0])
    # t=0, s=1, ss=1
    rdm2_red[1] = lib.einsum('sr, srpq -> pq', rho_pair0[0].conj(), rdm2[1])
    # t=1, s=1, ss=2
    rdm2_red[2] = lib.einsum('sr, srpq -> pq', rho_pair0[1].conj(), rdm2[2])
    # t=1, s=0, ss=3
    rdm2_red[3] = lib.einsum('sr, srpq -> pq', rho_pair0[1].conj(), rdm2[1].conj().transpose(2, 3, 0, 1))

    cf += lib.einsum('gqp, pq -> g', rho_pair[0], rdm2_red[0])
    cf += lib.einsum('gqp, pq -> g', rho_pair[1], rdm2_red[2])
    cf -= lib.einsum('gqp, pq -> g', rho_pair[1], rdm2_red[1])
    cf -= lib.einsum('gqp, pq -> g', rho_pair[0], rdm2_red[3])
    
    return cf

def get_rho_pair_q(ao_g, mo, q, nz, nxy, z_val):
    assert (mo.ndim == 3)
    ngrids = ao_g.shape[0]
    spin, nao, nmo = mo.shape
    assert(nz*nxy == ngrids)
    mo_g = np.zeros((spin, ngrids, nmo), dtype=np.complex128)
    for s in range(spin):
        mo_g[s] = ao_g.dot(mo[s])
    mo_g = mo_g.reshape((spin, nz, nxy, nmo))
    
    rho_sz = np.einsum('szai, szaj -> szji', mo_g.conj(), mo_g, optimize=True)
    qz = np.einsum('q, z -> qz', q, z_val)
    eiqz = np.exp(1j*qz)
    rho_q = np.einsum('szji, qz -> sqji', rho_sz, eiqz, optimize=True)
    return rho_q 

def eval_spin_corr_func_q(rho_q, rdm1, rdm2, factor):
    """
    Stucture factor of 1D system.
    """
    spin, nq, nmo, _ = rho_q.shape
    cf = np.zeros((nq,), dtype=np.complex128)
    
    # one-body terms
    rdg = np.zeros((spin, nq, nmo, nmo), np.complex128)
    for s in range(spin):
        rdg[s] = (rho_q[s].conj().transpose((0, 2, 1))).dot(rdm1[s])
        # finite size correction, add 1.0 in the end.
        #cf += np.einsum('qij, qji -> q', rdg[s], rho_q[s], optimize=True)
    
    for s in range(spin):
        tr2 = lib.einsum('qij, ji -> q', rho_q[s], rdm1[s])
        for t in range(spin):
            factor = (-1)**s * (-1)**t
            tr1 = np.sum(rdg[t], axis=(1, 2))
            cf -= (factor) * tr1 * tr2
    
    # two-body terms:
    rdm2_red = np.zeros((spin*spin, nq, nmo, nmo), np.complex128)
    # t=0, s=0, ss=0
    rdm2_red[0] = lib.einsum('Qsr, srpq -> Qpq', rho_q[0].conj(), rdm2[0])
    # t=0, s=1, ss=1
    rdm2_red[1] = lib.einsum('Qsr, srpq -> Qpq', rho_q[0].conj(), rdm2[1])
    # t=1, s=1, ss=2
    rdm2_red[2] = lib.einsum('Qsr, srpq -> Qpq', rho_q[1].conj(), rdm2[2])
    # t=1, s=0, ss=3
    rdm2_red[3] = lib.einsum('Qsr, srpq -> Qpq', rho_q[1].conj(), rdm2[1].conj().transpose(2, 3, 0, 1))

    cf += np.einsum('Qqp, Qpq -> Q', rho_q[0], rdm2_red[0], optimize=True)
    cf += np.einsum('Qqp, Qpq -> Q', rho_q[1], rdm2_red[2], optimize=True)
    cf -= np.einsum('Qqp, Qpq -> Q', rho_q[1], rdm2_red[1], optimize=True)
    cf -= np.einsum('Qqp, Qpq -> Q', rho_q[0], rdm2_red[3], optimize=True)

    return cf*factor + 1.0

def eval_spin_corr_func_lo(rdm1_lo, rdm2_lo, idx1, idx2):
    r"""
    Evaluate the spin correlation function based on LO indices.
    \sum_{i in idx1, j in idx2} <S_i S_j>
    
    Args:
        rdm1_lo: rdm1 in lo
        rdm2_lo: rdm2 in lo
        idx1: idx for the first atom
        idx2: idx for the second atom

    Returns:
        a float number for correlation function value.
    """
    rdm1_a, rdm1_b = rdm1_lo
    rdm2_aa, rdm2_ab, rdm2_bb = rdm2_lo
    norb = rdm1_a.shape[-1]
    mesh = np.ix_(idx1, idx1, idx2, idx2)

    delta = np.eye(norb)
    rdm1_a_delta = np.einsum('ij, kl -> ikjl', rdm1_a, delta)
    rdm1_b_delta = np.einsum('ij, kl -> ikjl', rdm1_b, delta)
    rdm1_tmp = rdm1_a_delta + rdm1_b_delta

    Az_iijj = 0.25 * (rdm1_tmp + rdm2_aa + rdm2_bb - rdm2_ab - rdm2_ab.transpose((2, 3, 0, 1)))
    Axy_iijj = 0.5 * (rdm1_tmp - rdm2_ab.transpose(0, 3, 1, 2) - rdm2_ab.transpose(1, 2, 0, 3))
    cf = np.einsum('iijj->', Az_iijj[mesh] + Axy_iijj[mesh])
    return cf

# ****************************************************************************
# plot density of states DOS
# ****************************************************************************

def get_dos(mo_energy, ndos=301, e_min=None, e_max=None, e_fermi=None, \
        sigma=0.005, mo_coeff=None):
    """
    Compute density of states for a given set of MOs (with kpts).
    If mo_coeff is None, the total (spin-)dos is calculated,
    Otherwise, orbital-based (spin-)pdos is calculated.
    DOS shape: ((spin,) ndos)
    PDOS shape: ((spin,), nlo, ndos)
    
    Args:
        mo_energy: ((spin,), nkpts, nmo)
        ndos: number of points to plot
        e_min: left boundary of plot range
        e_max: right boundary of plot range
        e_fermi: fermi level, if given shift the zero as fermi level.
        sigma: smearing value
        mo_coeff: C_lo_mo for character analysis (PDOS), 
                  shape ((spin,) nkpts, nlo, nmo)
        efermi
    
    Returns:
        elist: (ndos)
        dos: ((spin,), (nlo,), ndos)
    """
    mo_energy = np.asarray(mo_energy)
    if e_fermi is not None:
        mo_energy = mo_energy - e_fermi
    nkpts, nmo = mo_energy.shape[-2:]
    mo_energy_min = mo_energy.min()
    mo_energy_max = mo_energy.max()
    margin = max(10 * sigma, 0.05 * (mo_energy_max - mo_energy_min)) # margin
    if e_min is None:
        e_min = mo_energy_min - margin
    if e_max is None:
        e_max = mo_energy_max + margin
    elist = np.linspace(e_min, e_max, ndos)
    norm = sigma * np.sqrt(2 * np.pi)
    tsigma = 2.0 * sigma ** 2
    if mo_energy.ndim == 2:
        if mo_coeff is None: # total dos
            dos = np.zeros_like(elist)
            for i, e_curr in enumerate(elist):
                dos[i] = np.sum(np.exp(-((mo_energy-e_curr)**2) / tsigma))
        else: # pdos
            mo_coeff = np.asarray(mo_coeff)
            nao, nmo = mo_coeff.shape[-2:]
            dos = np.zeros((nao, ndos))
            # kpm, kpm -> kpm -> pkm
            mo_sq = (mo_coeff.conj() * mo_coeff).real.transpose(1, 0, 2)
            for i, e_curr in enumerate(elist):
                # pkm, km -> p
                dos[:, i] = np.sum(mo_sq * np.exp(-((mo_energy-e_curr)**2) \
                        / tsigma), axis=(1, 2))
    else:
        spin = mo_energy.shape[0]
        if mo_coeff is None:
            dos = np.zeros((spin,) + elist.shape)
            for s in range(spin):
                for i, e_curr in enumerate(elist):
                    dos[s, i] = np.sum(np.exp(-((mo_energy[s]-e_curr)**2) \
                            / tsigma))
        else:
            mo_coeff = np.asarray(mo_coeff)
            nao, nmo = mo_coeff.shape[-2:]
            dos = np.zeros((spin, nao) + elist.shape)
            # skpm, skpm -> skpm -> spkm
            mo_sq = (mo_coeff.conj() * mo_coeff).real.transpose(0, 2, 1, 3)
            for s in range(spin):
                for i, e_curr in enumerate(elist):
                    # pkm, km -> pkm -> p
                    dos[s, :, i] = np.sum(mo_sq[s] * np.exp(-((mo_energy[s] - \
                            e_curr)**2) / tsigma), axis=(1, 2))
    return elist, dos / (nkpts * norm)

def plot_dos(elist, pdos, idx_dic=None, color_dic=None, \
        fig_size=(12, 6), fig_name="pdos.pdf", unit='eV', text=None, **kwargs):
    """
    Plot (projected) density of states.
    
    Args:
        elist: energy range, shape (ndos,)
        pdos: density of states, shape ((nlo,), ndos)
        idx_dic: a dictionary required for pdos plotting, 
                 should be {"orbital name", idx}
        color_dic: a dictionary for pdos coloring,
                 should be {"orbital name", "color name"},
                 if provided, only plot the lines that have color.
        fig_size: size of figure, default is (12, 6)
        fig_name: figure name
        unit: unit of E in figure
        text: a label on the left upper corner.
    
    Returns:
        plt: matplotlib plot object.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    fig, ax = plt.subplots(figsize=fig_size)
    ax = plt.gca()

    if unit == 'eV':
        from pyscf.data.nist import HARTREE2EV
        elist = np.asarray(elist) * HARTREE2EV
    
    if pdos.ndim == 1: # restricted total DOS
        plt.plot(elist, pdos, label='total', color='grey', linewidth=1)
    elif pdos.ndim == 2 and (idx_dic is not None): # restricted PDOS
        dos = pdos.sum(axis=0)
        plt.plot(elist, dos, label='total', color='grey', linewidth=1)
        if color_dic is None:
            for orb_name, idx in idx_dic.items():
                pdos_i = pdos[idx].sum(axis=0)
                plt.plot(elist, pdos_i, label=orb_name, linewidth=1)
        else:
            for orb_name, idx in idx_dic.items():
                if orb_name in color_dic:
                    pdos_i = pdos[idx].sum(axis=0)
                    plt.plot(elist, pdos_i, label=orb_name, \
                            color=color_dic[orb_name], linewidth=1)
    elif pdos.ndim == 2 and (idx_dic is None): # unrestricted total DOS
        assert pdos.shape[0] == 2
        plt.plot(elist, pdos[0], label='total', color='grey', linewidth=1)
        plt.plot(elist, -pdos[1], color='grey', linewidth=1)
    elif pdos.ndim == 3: # unrestricted PDOS
        assert idx_dic is not None  
        dos = pdos.sum(axis=1)
        plt.plot(elist, dos[0], label='total', color='grey', linewidth=1)
        plt.plot(elist, -dos[1], color='grey', linewidth=1)
        if color_dic is None:
            for orb_name, idx in idx_dic.items():
                pdos_i = pdos[:, idx].sum(axis=1)
                tmp = plt.plot(elist, pdos_i[0], label=orb_name, \
                        linewidth=1)[0]
                plt.plot(elist, -pdos_i[1], color=tmp.get_color(), \
                        linewidth=1)
        else:
            for orb_name, idx in idx_dic.items():
                if orb_name in color_dic:
                    pdos_i = pdos[:, idx].sum(axis=1)
                    tmp = plt.plot(elist, pdos_i[0], label=orb_name, \
                            color=color_dic[orb_name], linewidth=1)[0]
                    plt.plot(elist, -pdos_i[1], color=tmp.get_color(), \
                            linewidth=1)
    else:
        raise ValueError("Unknown pdos shape %s" %(str(pdos.shape)))

    ax.legend(fancybox=False, framealpha=1.0, edgecolor='black', fontsize=10, \
            frameon=False, loc='upper right')
    
    # plot efermi line
    efermi_x = [0.0, 0.0]
    efermi_y = ax.get_ylim()
    plt.plot(efermi_x, efermi_y, linestyle='--', color='black', linewidth=1)
    ax.set_ylim(efermi_y)

    plt.xlabel("$E$ [%s]"%(unit), fontsize=10) 
    plt.ylabel("PDOS", fontsize=10) 
    if text is not None:
        plt.text(0.02, 0.96, text, horizontalalignment='left', \
                verticalalignment='center', transform=ax.transAxes, \
                fontsize=10)
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    return plt

def plot_bands(ax, kdis, ew, weights=None, cmap=None, linewidth=4):
    """
    Plot bands for given ax object.

    Args:
        ax:
        kdis: kpoint distance, (nkpts)
        ew: mo energy, (nkpts, nbands)
        weights: weights for color map, should be 0 - 1, (nkpts, nbands)
        cmap: colormap type.
        linewidth: line width.

    Returns:
        line: collection of lines.
    """
    from matplotlib.pyplot import Normalize
    from matplotlib.collections import LineCollection
    norm = Normalize(0.0, 1.0)
    kdis = np.asarray(kdis)
    ew   = np.asarray(ew)
    nbands = ew.shape[-1]
    if weights is None:
        weights = np.ones_like(ew)
    if cmap is None:
        cmap = 'Greys'

    for n in range(nbands):
        x = kdis
        y = ew[:, n]
        points = np.array((x, y)).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        dydx = weights[:, n]
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)

    return line
