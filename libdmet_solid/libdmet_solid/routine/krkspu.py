#!/usr/bin/env python

"""
Restricted DFT+U with kpoint sampling.
Based on KRHF routine.

Refs: PRB, 1998, 57, 1505.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import copy
import itertools as it
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.dft import krks
from pyscf.data.nist import HARTREE2EV

from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo import iao, lowdin_k
from libdmet_solid.utils import format_idx, mdot 

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, \
        kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + Hubbard U terms.

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : (nkpts, nao, nao) or (*, nkpts, nao, nao) ndarray
        Veff = J + Vxc + V_U.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts

    # J + V_xc
    vxc = krks.get_veff(ks, cell=cell, dm=dm, dm_last=dm_last, \
            vhf_last=vhf_last, hermi=hermi, kpts=kpts, kpts_band=kpts_band)
    
    # V_U
    nkpts = len(kpts)
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    rdm1_lo = make_basis.transform_rdm1_to_lo(dm, C_ao_lo, ovlp)
    E_U = 0.0
    weight = 1.0 / nkpts
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ", \
                    " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            U_mesh = np.ix_(idx, idx)
            P_loc = 0.0
            for k in range(nkpts):
                S_k = ovlp[k]
                C_k = C_ao_lo[k][:, idx]
                P_k = rdm1_lo[k][U_mesh]
                SC = np.dot(S_k, C_k)
                vxc[k] += mdot(SC, (np.eye(P_k.shape[-1]) - P_k) \
                        * (val * 0.5), SC.conj().T)
                E_U += (val * 0.5) * (P_k.trace() - np.dot(P_k, P_k).trace() * 0.5)
                P_loc += P_k
            P_loc = P_loc.real / nkpts
            logger.info(ks, "%s\n%s", lab_string, P_loc)
            logger.info(ks, "-" * 79)
    
    E_U *= weight 
    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U)
    return vxc
    
def energy_elec(ks, dm_kpts=None, h1e_kpts=None, vhf=None):
    """
    Electronic energy for KRKSpU.
    """
    if h1e_kpts is None: h1e_kpts = ks.get_hcore(ks.cell, ks.kpts)
    if dm_kpts is None: dm_kpts = ks.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.cell, dm_kpts)

    weight = 1.0 / len(h1e_kpts)
    e1 = weight * np.einsum('kij,kji', h1e_kpts, dm_kpts)
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    ks.scf_summary['e1'] = e1.real
    ks.scf_summary['coul'] = vhf.ecoul.real
    ks.scf_summary['exc'] = vhf.exc.real
    ks.scf_summary['E_U'] = vhf.E_U.real
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, vhf.ecoul,\
            vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

def set_U(ks, U_idx, U_val):
    """
    Regularize the U_idx and U_val to each atom,
    and set ks.U_idx, ks.U_val, ks.U_lab.
    """
    assert len(U_idx) == len(U_val)
    ks.U_val = []
    ks.U_idx = []
    ks.U_lab = []

    lo_labels = np.asarray(ks.cell.ao_labels())
    for i, idx in enumerate(U_idx):
        if isinstance(idx, str):
            lab_idx = ks.cell.search_ao_label(idx)
            labs = lo_labels[lab_idx]
            labs = zip(lab_idx, labs)
            for j, idxj in it.groupby(labs, key=lambda x: x[1].split()[0]):
                ks.U_idx.append(list(list(zip(*idxj))[0]))
                ks.U_val.append(U_val[i])
        else:
            ks.U_idx.append(copy.deepcopy(idx))
            ks.U_val.append(U_val[i])
    ks.U_val = np.asarray(ks.U_val) / HARTREE2EV
    logger.info(ks, "-" * 79)
    logger.debug(ks, 'U indices and values: ')
    for idx, val in zip(ks.U_idx, ks.U_val):
        ks.U_lab.append(lo_labels[idx])
        logger.debug(ks, '%6s [%.6g eV] ==> %-100s', format_idx(idx), \
                val * HARTREE2EV, "".join(lo_labels[idx]))
    logger.info(ks, "-" * 79)

def make_minao_lo(ks, minao_ref, pmol=None):
    """
    Construct minao local orbitals.
    """
    cell = ks.cell
    nao = cell.nao_nr()
    kpts = ks.kpts
    nkpts = len(kpts)
    ovlp = ks.get_ovlp()
    C_ao_minao, labels = iao.proj_ref_ao(cell, minao=minao_ref, \
                        kpts=kpts, pmol=pmol, return_labels=True)
    C_ao_minao = iao.vec_lowdin(C_ao_minao, ovlp)
    labels = np.asarray(labels)
    
    C_ao_lo = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    for idx, lab in zip(ks.U_idx, ks.U_lab):                    
        idx_minao = [i for i, l in enumerate(labels) if l in lab]
        assert len(idx_minao) == len(idx)
        C_ao_sub = C_ao_minao[:, :, idx_minao]
        C_ao_lo[:, :, idx] = C_ao_sub
    return C_ao_lo

class KRKSpU(krks.KRKS):
    """
    RKSpU class adapted for PBCs with k-point sampling.
    """
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), \
                 U_idx=[], U_val=[], C_ao_lo='minao', **kwargs):
        """
        DFT+U args:
            U_idx: can be 
                   list of list: each sublist is a set of LO indices to add U.
                   list of string: each string is one kind of LO orbitals, 
                                   e.g. ['Ni 3d', '1 O 2pz'], in this case,
                                   LO should be aranged as ao_labels order.
                   or a combination of these two.
            U_val: a list of effective U [in eV], i.e. U-J in Dudarev's DFT+U.
                   each U corresponds to one kind of LO orbitals, should have
                   the same length as U_idx.
            C_ao_lo: LO coefficients, can be 
                     np.array, shape ((spin,), nkpts, nao, nlo),
                     string, in 'minao', 'meta-lowdin', 'lowdin'.
                     default is 'minao'.
        
        Kwargs:
            minao_ref: reference for minao orbitals, default is 'MINAO'. 
            pmol: reference pmol for minao orbitals. default is None.
            pre_orth_ao: can be 
                         None: using ANO as reference basis for constructing 
                               (meta)-Lowdin C_ao_lo
                         otherwise use identity (AO) as reference.
        """
        try:
            krks.KRKS.__init__(self, cell, kpts, xc=xc, exxdiv=exxdiv)
        except TypeError:
            # backward compatibility
            krks.KRKS.__init__(self, cell, kpts)
            self.xc = xc
            self.exxdiv = exxdiv
        
        set_U(self, U_idx, U_val)
        
        if isinstance(C_ao_lo, str): 
            if C_ao_lo == 'minao':
                minao_ref = kwargs.get("minao_ref", "MINAO")
                pmol = kwargs.get("pmol", None)
                self.C_ao_lo = make_minao_lo(self, minao_ref, pmol)
            else: # (meta)-lowdin, w/ or w/o ref AO.
                pre_orth_ao = kwargs.get("pre_orth_ao", None)
                self.C_ao_lo = lowdin_k(self, method=C_ao_lo, pre_orth_ao=pre_orth_ao)
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)
        if self.C_ao_lo.ndim == 4:
            self.C_ao_lo = self.C_ao_lo[0]
        self._keys = self._keys.union(["U_idx", "U_val", "C_ao_lo", "U_lab"])

    get_veff = get_veff
    energy_elec = energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf.pbc import gto
    from libdmet_solid.system import lattice
    np.set_printoptions(3, linewidth=1000, suppress=True)
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.build()
    kmesh = [2, 1, 1]
    kpts = cell.make_kpts(kmesh, wrap_around=True) 
    Lat = lattice.Lattice(cell, kmesh)
    #U_idx = ["2p", "2s"]
    #U_val = [5.0, 2.0]
    U_idx = ["1 C 2p"]
    U_val = [5.0]
    
    mf = KRKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, C_ao_lo='minao',
            minao_ref='gth-szv')
    mf.conv_tol = 1e-10
    print (mf.U_idx)
    print (mf.U_val)
    print (mf.C_ao_lo.shape)
    print (mf.kernel())
    Lat.analyze(mf)

