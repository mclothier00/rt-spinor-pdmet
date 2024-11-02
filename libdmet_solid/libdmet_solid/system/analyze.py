#! /usr/bin/env python

"""
Analyze functions for kLO, kMO and rdm1.

Author:
    Zhihao Cui
"""

from pyscf.lib import logger as pyscf_logger
from pyscf.scf.hf import mulliken_pop as mulliken_pop_rhf

from libdmet_solid.basis_transform import make_basis 
from libdmet_solid.utils.misc import add_spin_dim
from libdmet_solid.system.fourier import *

def check_lo(lattice, C_ao_lo, kpts=None, ovlp=None, tol=1e-10):
    """
    Check whether a set of k-dependent local orbitals,
    whether has imaginary part or violates time reversal symmetry.
    If ovlp is not None, check the orthonormality as well.
    """
    log.info("-----------------------------------------------------------")
    log.info("Check the reality and time reversal symm of orbitals.")
    if kpts is None:
        kpts = lattice.kpts
    nkpts = len(kpts)
    kpts_scaled = lattice.cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    C_ao_lo = np.asarray(C_ao_lo)
    
    # check time reversal symmetry
    diff_k_mk = 0.0
    weights = np.ones(nkpts, dtype=np.int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    if C_ao_lo.ndim == 3:
                        diff_k_mk = max(diff_k_mk, max_abs(C_ao_lo[j] \
                                - C_ao_lo[i].conj()))
                    else:
                        for s in range(C_ao_lo.shape[0]):
                            diff_k_mk = max(diff_k_mk, max_abs(C_ao_lo[s, j] \
                                - C_ao_lo[s, i].conj()))
                    weights[i] = 2
                    weights[j] = 0
                    break
    log.info("Maximal difference between k and -k orbitals: %.2e", diff_k_mk)
    
    # check imaginary
    C_ao_lo_R = lattice.k2R(C_ao_lo)
    imag_norm = max_abs(C_ao_lo_R.imag)
    log.info("Imaginary part of orbitals: %.2e", imag_norm)
    
    if ovlp is not None:
        from libdmet_solid.lo.lowdin import check_orthonormal
        log.info("Orthonormal: %s", check_orthonormal(C_ao_lo, ovlp, tol=tol))
    log.info("-----------------------------------------------------------")
    return imag_norm, diff_k_mk

def symmetrize_lo(lattice, C_ao_lo, kpts=None, tol=1e-10, real_first=False):
    """
    Check whether a set of k-dependent local orbitals,
    whether has imaginary part or violates time reversal symmetry.
    """
    log.info("-----------------------------------------------------------")
    log.info("Impose the reality and time reversal symm of orbitals.")
    if kpts is None:
        kpts = lattice.kpts
    nkpts = len(kpts)
    kpts_scaled = lattice.cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    C_ao_lo_symm = np.array(C_ao_lo, copy=True)
    
    if real_first:
        # enforce reality 
        C_ao_lo_symm_R = lattice.k2R(C_ao_lo_symm)
        imag_norm = max_abs(C_ao_lo_symm_R.imag)
        C_ao_lo_symm = lattice.R2k(C_ao_lo_symm_R.real)
        log.info("Imaginary part of orbitals: %s", imag_norm)
    
    # enforce time reversal symmetry
    diff_k_mk = 0.0
    weights = np.ones(nkpts, dtype=np.int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    if C_ao_lo_symm.ndim == 3:
                        diff_k_mk = max(diff_k_mk, max_abs(C_ao_lo_symm[j] \
                                - C_ao_lo_symm[i].conj()))
                        C_ao_lo_symm[j] = C_ao_lo_symm[i].conj()
                    else:
                        for s in range(C_ao_lo_symm.shape[0]):
                            diff_k_mk = max(diff_k_mk, \
                                    max_abs(C_ao_lo_symm[s, j] - \
                                    C_ao_lo_symm[s, i].conj()))
                            C_ao_lo_symm[s, j] = C_ao_lo_symm[s, i].conj()
                    weights[i] = 2
                    weights[j] = 0
                    break
    log.info("Maximal difference between k and -k orbitals: %.2e", diff_k_mk)
    
    if not real_first:
        # enforce reality 
        C_ao_lo_symm_R = lattice.k2R(C_ao_lo_symm)
        imag_norm = max_abs(C_ao_lo_symm_R.imag)
        C_ao_lo_symm = lattice.R2k(C_ao_lo_symm_R.real)
        log.info("Imaginary part of orbitals: %.2e", imag_norm)
    log.info("-----------------------------------------------------------")
    return C_ao_lo_symm

def symmetrize_kmf(lattice, kmf, tol=1e-10):
    """
    Symmetrize kmf with time reversal symmetry.
    """
    from pyscf.scf import uhf
    is_uhf = isinstance(kmf, uhf.UHF)
    kpts = lattice.kpts
    nkpts = len(kpts)
    kpts_scaled = lattice.cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    weights = np.ones(nkpts, dtype=np.int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    if not is_uhf:
                        kmf.mo_coeff[i]  = kmf.mo_coeff[j].conj()
                        kmf.mo_energy[i] = kmf.mo_energy[j]
                        kmf.mo_occ[i] = kmf.mo_occ[j]
                    else:
                        for s in range(2):
                            kmf.mo_coeff[s][i]  = kmf.mo_coeff[s][j].conj()
                            kmf.mo_energy[s][i] = kmf.mo_energy[s][j]
                            kmf.mo_occ[s][i] = kmf.mo_occ[s][j]
                    weights[i] = 2
                    weights[j] = 0
                    break
    return kmf

def analyze(lattice, kmf, C_ao_lo=None, labels=None, \
        verbose=pyscf_logger.DEBUG, rdm1_lo_R0=None, method='meta-lowdin'):
    """
    Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment

    Args:
        lattice: lattice object.
        kmf: kscf object.
        C_ao_lo: shape ((spin,), nkpts, nao, nlo), 
                 if None, meta Lowdin is used.
        labels: LO labels, list of strings.
        verbose: verbose level.
        rdm1_lo_R0: analyze a given rdm1_lo at R0.
        method: default LO method.
    """
    from libdmet_solid.lo import lowdin
    kmf.dump_scf_summary(verbose)
    ovlp = np.asarray(kmf.get_ovlp())
    
    if rdm1_lo_R0 is None:
        mo_occ   = kmf.mo_occ
        mo_coeff = kmf.mo_coeff
        rdm1 = np.asarray(kmf.make_rdm1(mo_coeff, mo_occ))
    else:
        rdm1 = None

    if C_ao_lo is None:
        C_ao_lo = lowdin.lowdin_k(kmf, method=method, s=ovlp)
    return mulliken_lo(lattice, rdm1, ovlp, C_ao_lo=C_ao_lo, labels=labels, \
            verbose=verbose, rdm1_lo_R0=rdm1_lo_R0)

def mulliken_lo(lattice, rdm1, ovlp, C_ao_lo, labels, \
        verbose=pyscf_logger.DEBUG, rdm1_lo_R0=None, rdm1_lo_R0_2=None):
    """
    A modified Mulliken population analysis, based on given LO.
    """
    cell = lattice.cell.copy()
    log = pyscf_logger.new_logger(cell, verbose)
    C_ao_lo = np.asarray(C_ao_lo)
    
    if rdm1_lo_R0 is None:
        rdm1_lo = lattice.k2R(make_basis.transform_rdm1_to_lo(rdm1, C_ao_lo, \
                ovlp))
        if rdm1_lo.ndim == 3: # RHF
            rdm1_lo_R0 = rdm1_lo[0]
        else:
            rdm1_lo_R0 = rdm1_lo[:, 0]
    rdm1_lo_R0 = np.asarray(rdm1_lo_R0)
    nlo = rdm1_lo_R0.shape[-1]
    
    if labels is None:
        idx_to_ao_labels = np.arange(nlo)
    else:
        # IAO indices need to resort according to each atom.
        atom_ids = [int(lab.split()[0]) for lab in labels]
        idx_to_ao_labels = np.argsort(atom_ids, kind='mergesort')
        
        labels_ao = [labels[idx] for idx in idx_to_ao_labels]
        # a hack, only keep atom id
        labels_ao_no_fmt = [(int(lab.split()[0]),) for lab in labels_ao] 

        def ao_labels(fmt=True):
            if fmt:
                return labels_ao
            else:
                return labels_ao_no_fmt
        cell.ao_labels = ao_labels

    # resort rdm1_lo_R0 according to LO labels, 
    # so that the order is the same as AOs.
    mesh = np.ix_(idx_to_ao_labels, idx_to_ao_labels) 
    if rdm1_lo_R0.ndim == 2: # RHF
        rdm1_lo_R0 = rdm1_lo_R0[mesh]
    else:
        if rdm1_lo_R0.shape[0] == 1: # RHF
            rdm1_lo_R0 = rdm1_lo_R0[0][mesh]
        else: # UHF
            rdm1_lo_R0 = np.asarray([rdm1_lo_R0[s][mesh] \
                    for s in range(rdm1_lo_R0.shape[0])])

    log.note(' ** Mulliken pop on LOs **')
    if rdm1_lo_R0_2 is None:
        if rdm1_lo_R0.ndim == 2:
            return mulliken_pop_rhf(cell, rdm1_lo_R0, np.eye(nlo), log)
        else:
            return mulliken_pop_uhf(cell, rdm1_lo_R0, np.eye(nlo), log)
    else:
        if rdm1_lo_R0_2.ndim == 2: # RHF
            rdm1_lo_R0_2 = rdm1_lo_R0_2[mesh]
        else:
            if rdm1_lo_R0_2.shape[0] == 1: # RHF
                rdm1_lo_R0_2 = rdm1_lo_R0_2[0][mesh]
            else: # UHF
                rdm1_lo_R0_2 = np.asarray([rdm1_lo_R0_2[s][mesh] \
                        for s in range(rdm1_lo_R0_2.shape[0])])
        compare_density(cell, rdm1_lo_R0, rdm1_lo_R0_2, np.eye(nlo))
        return None, None

def mulliken_lo_R0(lattice, rdm1_lo_R0, rdm1_lo_R0_2=None, labels=None):
    return mulliken_lo(lattice, None, None, None, labels, \
            rdm1_lo_R0=rdm1_lo_R0, rdm1_lo_R0_2=rdm1_lo_R0_2)

def mulliken_pop_uhf(mol, dm, s=None, verbose=pyscf_logger.DEBUG):
    """
    Mulliken population analysis, UHF case.
    Include local magnetic moment.
    """
    if s is None: s = hf.get_ovlp(mol)
    log = pyscf_logger.new_logger(mol, verbose)
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    pop_a = np.einsum('ij,ji->i', dm[0], s).real
    pop_b = np.einsum('ij,ji->i', dm[1], s).real

    log.info(' ** Mulliken pop            alpha | beta      %12s **'%("magnetism"))
    for i, s in enumerate(mol.ao_labels()):
        log.info('pop of  %-14s %10.5f | %-10.5f  %10.5f',
                 s.strip(), pop_a[i], pop_b[i], pop_a[i] - pop_b[i])
    log.info('In total               %10.5f | %-10.5f  %10.5f', \
            sum(pop_a), sum(pop_b), sum(pop_a) - sum(pop_b))

    log.note(' ** Mulliken atomic charges    ( Nelec_alpha | Nelec_beta )'
            ' %12s **'%("magnetism"))
    nelec_a = np.zeros(mol.natm)
    nelec_b = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        nelec_a[s[0]] += pop_a[i]
        nelec_b[s[0]] += pop_b[i]
    chg = mol.atom_charges() - (nelec_a + nelec_b)
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note('charge of %4d%s =   %10.5f  (  %10.5f   %10.5f )   %10.5f',
                 ia, symb, chg[ia], nelec_a[ia], nelec_b[ia], nelec_a[ia] - nelec_b[ia])
    return (pop_a,pop_b), chg

def compare_density(mol, rdm_1, rdm_2, s=None, verbose=pyscf_logger.DEBUG):
    r"""
    Compare two density matrices.
    """
    log = pyscf_logger.new_logger(mol, verbose)
    if s is None:
        s = get_ovlp(mol)
    rdm_1 = np.asarray(rdm_1)
    rdm_2 = np.asarray(rdm_2)
    assert rdm_1.shape == rdm_2.shape
    if rdm_1.ndim == 2: # RHF
        pop1 = np.einsum('ij,ji->i', rdm_1, s).real
        pop2 = np.einsum('ij,ji->i', rdm_2, s).real
        log.info(' ** Mulliken pop   %10s  %10s **', "sys1", "sys2")
        for i, s in enumerate(mol.ao_labels()):
            log.info('pop of  %s %10.5f  %10.5f', s, pop1[i], pop2[i])

        log.note(' ** Mulliken atomic charges  **')
        chg1 = np.zeros(mol.natm)
        chg2 = np.zeros(mol.natm)
        for i, s in enumerate(mol.ao_labels(fmt=None)):
            chg1[s[0]] += pop1[i]
            chg2[s[0]] += pop2[i]
        chg1 = mol.atom_charges() - chg1
        chg2 = mol.atom_charges() - chg2
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            log.note('charge of  %d%s =   %10.5f  %10.5f', ia, symb, \
                    chg1[ia], chg2[ia])
    else: # ROHF, UHF
        pop1_a = np.einsum('ij,ji->i', rdm_1[0], s).real
        pop1_b = np.einsum('ij,ji->i', rdm_1[1], s).real
        pop2_a = np.einsum('ij,ji->i', rdm_2[0], s).real
        pop2_b = np.einsum('ij,ji->i', rdm_2[1], s).real
        log.info(" ** Mulliken pop        alpha | beta           "
                "  alpha | beta")
        for i, s in enumerate(mol.ao_labels()):
            log.info('pop of  %s %10.5f | %-10.5f  %10.5f | %-10.5f',
                     s, pop1_a[i], pop1_b[i], pop2_a[i], pop2_b[i])
        log.info('In total           %10.5f | %-10.5f  %10.5f | %-10.5f', \
                sum(pop1_a), sum(pop1_b), sum(pop2_a), sum(pop2_b))

        log.note(' ** Mulliken atomic charges   ( Nelec_alpha | Nelec_beta )'
                '     charges   ( Nelec_alpha | Nelec_beta ) **')
        nelec1_a = np.zeros(mol.natm)
        nelec1_b = np.zeros(mol.natm)
        nelec2_a = np.zeros(mol.natm)
        nelec2_b = np.zeros(mol.natm)
        for i, s in enumerate(mol.ao_labels(fmt=None)):
            nelec1_a[s[0]] += pop1_a[i]
            nelec1_b[s[0]] += pop1_b[i]
            nelec2_a[s[0]] += pop2_a[i]
            nelec2_b[s[0]] += pop2_b[i]
        chg1 = mol.atom_charges() - (nelec1_a + nelec1_b)
        chg2 = mol.atom_charges() - (nelec2_a + nelec2_b)
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            log.note('charge of  %d%s =   %10.5f  (  %10.5f   %10.5f )' \
                    '   %10.5f  (  %10.5f   %10.5f )',
                     ia, symb, chg1[ia], nelec1_a[ia], nelec1_b[ia], \
                               chg2[ia], nelec2_a[ia], nelec2_b[ia])

def analyze_kmo(kmf, C_ao_lo=None, lo_labels=None, C_lo_mo=None, num_max=4, \
        k_loop_first=True, mo_print_list=None, kpts_print_list=None, \
        nmo_print=None, nkpts_print=None):
    """
    Analyze k-MO at each k point for each MO.

    Args:
        mo_coeff: C_ao_mo
        C_ao_lo: C_ao_lo
        ovlp: AO overlap
        lo_labels: can be list or dict
        C_lo_mo: directly give C_lo_mo
        num_max: largest num_max component will be printed.
        k_loop_first: print the results at each kpts.
        mo_print_list: a list of MO indices to print.
        kpts_print_list: a list of kpt indices to print.
        nmo_print: number of MO to print
        nkpts_print: number of kpts to print.
        mo_energy: if provide, will print the energy range of a band.

    Returns:
        order: the order of lo component.
    """
    from libdmet_solid.lo import lowdin
    log.info("Analyze k-MO")
    log.info("-----------------------------------------------------------")
    
    mo_coeff = np.asarray(kmf.mo_coeff)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[None]
    spin, nkpts, nao, nmo = mo_coeff.shape
    mo_energy = np.asarray(kmf.mo_energy).reshape(spin, nkpts, nmo)
    mo_energy_min = np.amin(mo_energy, axis=1)
    mo_energy_max = np.amax(mo_energy, axis=1)
    ovlp = np.asarray(kmf.get_ovlp())

    if C_ao_lo is None:
        C_ao_lo = lowdin.lowdin_k(kmf, method='meta_lowdin', s=ovlp)
    C_ao_lo = np.asarray(C_ao_lo)
    C_ao_lo = add_spin_dim(C_ao_lo, spin)
    nlo = C_ao_lo.shape[-1]
    if C_lo_mo is None:
        C_lo_mo = make_basis.get_mo_ovlp_k(C_ao_lo, mo_coeff, ovlp)
    C_lo_mo_abs = np.abs(C_lo_mo) ** 2 * 100.0
    if lo_labels is None:
        lo_labels = kmf.cell.ao_labels()
    
    if nmo_print is None:
        nmo_print = nmo
    if nkpts_print is None:
        nkpts_print = nkpts
    if mo_print_list is None:
        mo_print_list = range(nmo_print)
    if kpts_print_list is None:
        kpts_print_list = range(nkpts_print)

    if isinstance(lo_labels, dict):
        lo_keys = list(lo_labels.keys())
        nlo_grouped = len(lo_keys)
        C_lo_mo_abs_grouped = np.empty((spin, nkpts, nlo_grouped, nmo))
        for l in range(nlo_grouped):
            C_lo_mo_abs_grouped[:, :, l] = C_lo_mo_abs[:, :, \
                    lo_labels[lo_keys[l]]].sum(axis=2)
        C_lo_mo_abs = C_lo_mo_abs_grouped
    else:
        assert len(lo_labels) == nlo
        lo_keys = np.asarray(lo_labels)

    order = np.argsort(C_lo_mo_abs, axis=2, kind='mergesort')[:, :, ::-1]
    for s in range(spin):
        log.info("spin sector: %s", s)
        if k_loop_first:
            for k in kpts_print_list:
                log.info(" kpt: %4s", k)
                for m in mo_print_list:
                    idx = order[s, k, :num_max, m]
                    string = "".join(["%15s (%5.1f) "%(lo_keys[id].strip(), \
                            C_lo_mo_abs[s, k, id, m]) for id in idx])
                    log.info("   MO %4s : %s", m, string)
        else:
            for m in mo_print_list:
                log.info(" MO: %4s    E: [%12.6f, %12.6f]", m, \
                        mo_energy_min[s, m], mo_energy_max[s, m])
                for k in kpts_print_list:
                    idx = order[s, k, :num_max, m]
                    string = "".join(["%15s (%5.1f) "%(lo_keys[id].strip(), \
                            C_lo_mo_abs[s, k, id, m]) for id in idx])
                    log.info("   kpt %4s : %s", k, string)
        log.info("-----------------------------------------------------------")
    return order

def get_symm_orb(mol, idx, perm_idx=None, tol=1e-6, ignore_empty_irep=True):
    """
    Get symmetrized orbitals with selected indices.

    Args:
        mol: cluster with symmetry.
        idx: selected indices.
        perm: permutation indices for AO axis, useful for find IAO symmetry.
        tol: tolerance for removing symmetrized orbitals.
        ignore_empty_irep: ignore the empty irreps.

    Returns:
        symm_orb: a list of symmetrized orbitals.
    """
    log.info("top symmetry: %s",  mol.topgroup)
    log.info("real symmetry: %s", mol.groupname)
    log.info("selected indices: %s", format_idx(idx))
    log.debug(0, "labels:\n%s", np.array(mol.ao_labels())[idx])
    
    if perm_idx is None:
        perm_idx = np.arange(len(idx))
    log.info("perm indices: %s", format_idx(perm_idx))
    log.debug(0, "labels after permutation:\n%s", \
            np.array(mol.ao_labels())[idx][perm_idx])

    norb_tot = 0
    nirep = 0
    irep_sizes = []
    symm_orb = []
    for i in range(len(mol.symm_orb)):
        tmp = mol.symm_orb[i][idx]
        norm = la.norm(tmp, axis=0)
        idx_non_zero = norm > tol
        tmp = tmp[:, idx_non_zero]
        
        if tmp.size > 0:
            orbs = tmp[:, [0]]
            for j in range(1, tmp.shape[-1]):
                res = orbs.T.dot(tmp[:, [j]])
                val = la.svd(res)[1][0]
                if val < tol:
                    orbs = np.hstack((orbs, tmp[:, [j]] / la.norm(tmp[:, [j]])))
            symm_orb.append(orbs[perm_idx])
            norb_tot += orbs.shape[-1]
            nirep += 1
            irep_sizes.append(orbs.shape[-1])
        else:
            orbs = tmp
            if not ignore_empty_irep:
                symm_orb.append(orbs)
                nirep += 1
                irep_sizes.append(orbs.shape[-1])
    
    log.info("nirep: %s", nirep)
    log.info("irep sizes: \n%s", np.array(irep_sizes))
    assert norb_tot == len(idx)
    return symm_orb
