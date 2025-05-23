#! /usr/bin/env python

import os
import h5py
import itertools as it
import numpy as np
from libdmet_solid.utils import logger as log
import libdmet_solid.utils.misc as misc
from libdmet_solid.utils.misc import eri_idx

def save(ints, fname="integral.h5"):
    """
    Save integrals into h5 file.
    """
    fint = h5py.File(fname, "w")
    fint["norb"]       = ints.norb
    fint["restricted"] = ints.restricted
    fint["bogoliubov"] = ints.bogoliubov
    fint["H0"]         = np.asarray(ints.H0)
    fint["H1_cd"]      = np.asarray(ints.H1["cd"])
    fint["H2_ccdd"]    = np.asarray(ints.H2["ccdd"])
    fint["ovlp"]       = np.asarray(ints.ovlp)
    if ints.bogoliubov:
        fint["H1_cc"]   = np.asarray(ints.H1["cc"])
        fint["H2_cccd"] = np.asarray(ints.H2["cccd"])
        fint["H2_cccc"] = np.asarray(ints.H2["cccc"])
    fint.close()

def load(fname="integral.h5"):
    """
    Load integrals from h5 file.
    """
    norb, restricted, bogoliubov, H0, H1, H2, ovlp = _load_integral(fname)
    return Integral(norb, restricted, bogoliubov, H0, H1, H2, ovlp=ovlp)

def _load_integral(fname):
    fint = h5py.File(fname, "r")
    norb       = int(fint["norb"][()])
    restricted = bool(fint["restricted"][()])
    bogoliubov = bool(fint["bogoliubov"][()])
    H0         = np.asarray(fint["H0"])
    H1         = {"cd": np.asarray(fint["H1_cd"])}
    H2         = {"ccdd": np.asarray(fint["H2_ccdd"])}
    ovlp       = np.asarray(fint["ovlp"])
    if bogoliubov:
        H1["cc"]   = np.asarray(fint["H1_cc"])
        H2["cccd"] = np.asarray(fint["H2_cccd"])
        H2["cccc"] = np.asarray(fint["H2_cccc"])
    fint.close()
    return (norb, restricted, bogoliubov, H0, H1, H2, ovlp)

class Integral(object):
    def __init__(self, norb, restricted, bogoliubov, H0, H1, H2, ovlp=None):
        self.norb = norb
        self.restricted = restricted
        self.bogoliubov = bogoliubov
        self.H0 = H0
        for key in H1:
            # H1 should has shape (spin, norb, norb)
            log.eassert(H1[key] is None or H1[key].ndim == 3, \
                    "invalid shape %s" %(str(H1[key].shape)))
        self.H1 = H1
        for key in H2:
            if H2[key] is not None:
                # H2 shape: (spin,) + (nao,)*4 or (pair,)*2 or (pair_pair,)*1
                length = H2[key].ndim
                log.eassert(length == 5 or length == 3 or length == 2, \
                        "invalid H2 shape: %s", str(H2[key].shape))
        self.H2 = H2

        # overlap
        # ZHC NOTE FIXME consider the ovlp is different for different spin?
        if ovlp is None:
            self.ovlp = np.eye(self.norb)
        else:
            self.ovlp = ovlp
    
    save = save

    def load(self, fname="integral.h5"):
        """
        Load integrals from h5 file.
        """
        norb, restricted, bogoliubov, H0, H1, H2, ovlp = _load_integral(fname)
        self.__init__(norb, restricted, bogoliubov, H0, H1, H2, ovlp=ovlp)

    def pairNoSymm(self):
        return list(it.product(range(self.norb), repeat=2))

    def pairSymm(self):
        return list(it.combinations_with_replacement(range(self.norb)[::-1], 2))[::-1]

    def pairAntiSymm(self):
        return list(it.combinations(range(self.norb)[::-1], 2))[::-1]

def dumpFCIDUMP(filename, integral, thr=1e-12):
    header = []
    if integral.bogoliubov:
        header.append(" &BCS NORB= %d," % integral.norb)
    else:
        header.append(" &FCI NORB= %d,NELEC= %d,MS2= %d," % (integral.norb, integral.norb, 0))
    header.append("  ORBSYM=" + "1," * integral.norb)
    header.append("  ISYM=1,")
    if not integral.restricted:
        header.append("  IUHF=1,")
    header.append(" &END")
    
    # ZHC NOTE ccdd can have permutation symmetry
    eri_format = get_eri_format(integral.H2["ccdd"], integral.norb)[0]
    def IDX(i, j, k, l):
        return eri_idx(i, j, k, l, integral.norb, eri_format)
        
    def writeInt(fout, val, i, j, k=-1, l=-1):
        if abs(val) > thr:
            fout.write("%20.16f%4d%4d%4d%4d\n" % (val, i+1, j+1, k+1, l+1))
    
    def insert_ccdd(fout, matrix, t, symm_herm=True, symm_spin=True):
        if symm_herm:
            p = integral.pairSymm()
        else:
            p = integral.pairNoSymm()
        
        if symm_spin:
            for (i, j), (k, l) in list(it.combinations_with_replacement(p[::-1], 2))[::-1]:
                writeInt(fout, matrix[t][IDX(i, j, k, l)], i, j, k, l)
        else:
            for (i, j), (k, l) in it.product(p, repeat=2):
                writeInt(fout, matrix[t][IDX(i, j, k, l)], i, j, k, l)
    
    def insert_cccd(fout, matrix, t):
        for (i, j), (k, l) in it.product(integral.pairAntiSymm(), integral.pairNoSymm()):
            writeInt(fout, matrix[t, i, j, k, l], i, j, k, l)

    def insert_cccc(fout, matrix, t, symm_spin=True):
        if symm_spin:
            for (i, j), (k, l) in list(it.combinations_with_replacement(integral.pairAntiSymm()[::-1], 2))[::-1]:
                writeInt(fout, matrix[t, i, j, l, k], i, j, k, l)
        else:
            for (i, j), (k, l) in it.product(integral.pairAntiSymm(), repeat=2):
                writeInt(fout, matrix[t, i, j, l, k], i, j, k, l)

    def insert_2dArray(fout, matrix, t, symm_herm=True):
        if symm_herm:
            for i, j in integral.pairSymm():
                writeInt(fout, matrix[t, i, j], i, j)
        else:
            for i, j in integral.pairNoSymm():
                writeInt(fout, matrix[t, i, j], i, j)

    def insert_H0(fout, val=0):
        fout.write("%20.16f%4d%4d%4d%4d\n" % (val, 0, 0, 0, 0)) # cannot be ignored even if smaller than thr

    if isinstance(filename, str):
        f = open(filename, "w", 1024*1024*128)
    elif isinstance(filename, file):
        f = filename

    f.write("\n".join(header) + "\n")
    
    if integral.restricted and not integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"], 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f, integral.H0)
    elif not integral.restricted and not integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"], 0)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 1)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 2, symm_spin=False)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 1)
        insert_H0(f, 0)
        insert_H0(f, integral.H0)
    elif integral.restricted and integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"], 0)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"], 0)
        insert_H0(f, 0)
        insert_cccc(f, integral.H2["cccc"], 0)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f,0)
        insert_2dArray(f, integral.H1["cc"], 0)
        insert_H0(f,0)
        insert_H0(f, integral.H0)
    else:
        insert_ccdd(f, integral.H2["ccdd"], 0, symm_herm=False)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 1, symm_herm=False)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 2, symm_herm=False, symm_spin=False)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"], 0)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"], 1)
        insert_H0(f, 0)
        insert_cccc(f, integral.H2["cccc"], 0, symm_spin=False)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 1)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cc"], 0, symm_herm=False)
        insert_H0(f, 0)
        insert_H0(f, integral.H0)

    if isinstance(filename, str):
        f.close()

def dumpFCIDUMP_no_perm(filename, integral, thr=1e-12):
    """
    Dump a FCIDUMP with symmetry of hermitian and (possible spin):
    * herm: (ij|kl) = (ji|lk)
    * spin: (ij|kl) = (kl|ij)
    * herm and spin: (ij|kl) = (ji|lk) = (kl|ij) = (lk|ji) # 4-fold
    NOTE that this 4-fold is different to pyscf's 4-fold:
    (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) [herm is already implicitly included]
    """
    # ZHC: FIXME TODO to support BCS integrals.
    # Seems have to dump all integrals, related to how block restore the integral.

    header = []
    if integral.bogoliubov:
        raise NotImplementedError
        header.append(" &BCS NORB= %d," % integral.norb)
    else:
        header.append(" &FCI NORB= %d,NELEC= %d,MS2= %d," % (integral.norb, integral.norb, 0))
    header.append("  ORBSYM=" + "1," * integral.norb)
    header.append("  ISYM=1,")
    if not integral.restricted:
        header.append("  IUHF=1,")
    header.append(" &END")

    def check_ind(ind_set, i, j, k, l, symm_spin=True):
        if symm_spin: # h + s, 4-fold
            return  ((j, i, l, k) not in ind_set) and \
                    ((l, k, j, i) not in ind_set)
        else: # h 
            return  ((j, i, l, k) not in ind_set)

    def writeInt(fout, val, i, j, k=-1, l=-1):
        if abs(val) > thr:
            fout.write("%20.16f%4d%4d%4d%4d\n" % (val, i+1, j+1, k+1, l+1))
    
    def insert_ccdd(fout, matrix, t, symm_herm=True, symm_spin=True):
        # NO permutation symmetry
        p = integral.pairNoSymm()
        ind_set = set()
        
        if symm_spin:
            for (i,j), (k,l) in reversed(list(it.combinations_with_replacement(reversed(p), 2))):
                #if check_ind(ind_set, i, j, k, l, symm_spin=symm_spin):
                    writeInt(fout, matrix[t, i, j, k, l], i, j, k, l)
                #    ind_set.add((i, j, k, l))
        else:
            for (i,j), (k,l) in it.product(p, repeat=2):
                #if check_ind(ind_set, i, j, k, l, symm_spin=symm_spin):
                    writeInt(fout, matrix[t, i, j, k, l], i, j, k, l)
                #    ind_set.add((i, j, k, l))
    
    def insert_cccd(fout, matrix, t):
        for (i,j), (k,l) in it.product(integral.pairAntiSymm(), integral.pairNoSymm()):
            writeInt(fout, matrix[t, i, j, k, l], i, j, k, l)

    def insert_cccc(fout, matrix, t, symm_spin=True):
        if symm_spin:
            for (i,j), (k,l) in list(it.combinations_with_replacement(integral.pairAntiSymm()[::-1], 2))[::-1]:
                writeInt(fout, matrix[t, i, j, l, k], i, j, k, l)
        else:
            for (i,j), (k,l) in it.product(integral.pairAntiSymm(), repeat=2):
                writeInt(fout, matrix[t, i, j, l, k], i, j, k, l)

    def insert_2dArray(fout, matrix, t, symm_herm=True):
        if symm_herm:
            for i,j in integral.pairSymm():
                writeInt(fout, matrix[t, i, j], i, j)
        else:
            for i,j in integral.pairNoSymm():
                writeInt(fout, matrix[t, i, j], i, j)

    def insert_H0(fout, val=0):
        fout.write("%20.16f%4d%4d%4d%4d\n" % (val, 0, 0, 0, 0)) # cannot be ignored even if smaller than thr

    if isinstance(filename, str):
        f = open(filename, "w", 1024*1024*128)
    elif isinstance(filename, file):
        f = filename

    f.write("\n".join(header) + "\n")
    if integral.restricted and not integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"], 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f, integral.H0)
    elif not integral.restricted and not integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"], 0)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 1)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 2, symm_spin=False)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 1)
        insert_H0(f, 0)
        insert_H0(f, integral.H0)
    elif integral.restricted and integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"], 0)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"], 0)
        insert_H0(f, 0)
        insert_cccc(f, integral.H2["cccc"], 0)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f,0)
        insert_2dArray(f, integral.H1["cc"], 0)
        insert_H0(f,0)
        insert_H0(f, integral.H0)
    else:
        insert_ccdd(f, integral.H2["ccdd"], 0, symm_herm=False)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 1, symm_herm=False)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccdd"], 2, symm_herm=False, symm_spin=False)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"], 0)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"], 1)
        insert_H0(f, 0)
        insert_cccc(f, integral.H2["cccc"], 0, symm_spin=False)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 0)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"], 1)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cc"], 0, symm_herm=False)
        insert_H0(f, 0)
        insert_H0(f, integral.H0)

    if isinstance(filename, str):
        f.close()

def readFCIDUMP(filename, norb, restricted, bogoliubov):
    with open(filename, "r") as f:
        head = f.readline()
        log.eassert((bogoliubov and "&BCS" in head) or (not bogoliubov and "&FCI" in head), \
            "particle number conservation is not consistent")
        log.eassert(norb == int(head.split(',')[0].split('=')[1]), "orbital number is not consistent")
        IUHF = False
        line = f.readline()
        while not "&END" in line and not "/" in line:
          IUHF = IUHF or "IUHF" in line
          line = f.readline()
        log.eassert(restricted != IUHF, "spin restriction is not consistent")
        if restricted and not bogoliubov:
            H0 = 0
            H1 = {"cd": np.zeros((1, norb, norb))}
            H2 = {"ccdd": np.zeros((1, norb, norb, norb, norb))}
            lines = f.readlines()
            for line in lines:
                tokens = line.split()
                val = float(tokens[0])
                i,j,k,l = [int(x) - 1 for x in tokens[1:]]
                if k >= 0 and l >= 0:
                    H2["ccdd"][0, i, j, k, l] = H2["ccdd"][0, j, i, k, l] = H2["ccdd"][0, i, j, l, k] = \
                        H2["ccdd"][0, j, i, l, k] = H2["ccdd"][0, k, l, i, j] = H2["ccdd"][0, k, l, j, i] = \
                        H2["ccdd"][0, l, k, i, j] = H2["ccdd"][0, l, k, j, i] = val
                elif i >= 0 and j >= 0:
                    H1["cd"][0, i, j] = H1["cd"][0, j, i] = val
                else:
                    H0 += val
        elif not restricted and not bogoliubov:
            H0 = 0
            H1 = {"cd": np.zeros((2,norb, norb))}
            H2 = {
                "ccdd": np.zeros((3,norb, norb, norb, norb)),
            }
            lines = f.readlines()
            section = 0
            for line in lines:
                tokens = line.split()
                val = float(tokens[0])
                i,j,k,l = [int(x) - 1 for x in tokens[1:]]
                if i < 0 and j < 0 and k < 0 and l < 0:
                    section += 1
                    H0 += val
                elif section == 0 or section == 1:
                    key = "ccdd"
                    H2[key][section,i,j,k,l] = H2[key][section,j,i,k,l] = H2[key][section,i,j,l,k] = \
                        H2[key][section,j,i,l,k] = H2[key][section,k,l,i,j] = H2[key][section,k,l,j,i] = \
                        H2[key][section,l,k,i,j] = H2[key][section,l,k,j,i] = val
                elif section == 2:
                    key = "ccdd"
                    H2[key][2,i,j,k,l] = H2[key][2,j,i,k,l] = H2[key][2,i,j,l,k] = \
                        H2[key][2,j,i,l,k] = val # cannot swap ij <-> kl
                elif section == 3 or section == 4:
                    log.eassert(k == -1 and l == -1, "Integral Syntax unrecognized")
                    H1["cd"][section-3,i,j] = H1["cd"][section-3,j,i] = val
        elif restricted and bogoliubov:
            H0 = 0
            H1 = {"cd": np.zeros((1,norb, norb)), "cc": np.zeros((1,norb, norb))}
            H2 = {
                "ccdd": np.zeros((1,norb, norb, norb, norb)),
                "cccd": np.zeros((1,norb, norb, norb, norb)),
                "cccc": np.zeros((1,norb, norb, norb, norb))
            }
            lines = f.readlines()
            section = 0
            for line in lines:
              tokens = line.split()
              val = float(tokens[0])
              i,j,k,l = [int(x) - 1 for x in tokens[1:]]
              if i < 0 and j < 0 and k < 0 and l < 0:
                  section += 1
                  H0 += val
              elif section == 0:
                  H2["ccdd"][0,i,j,k,l] = H2["ccdd"][0,j,i,k,l] = H2["ccdd"][0,i,j,l,k] = \
                    H2["ccdd"][0,j,i,l,k] = H2["ccdd"][0,k,l,i,j] = H2["ccdd"][0,k,l,j,i] = \
                    H2["ccdd"][0,l,k,i,j] = H2["ccdd"][0,l,k,j,i] = val
              elif section == 1:
                  H2["cccd"][0,i,j,k,l] = val
                  H2["cccd"][0,j,i,k,l] = -val
              elif section == 2:
                  H2["cccc"][0,i,j,l,k] = H2["cccc"][0,j,i,k,l] = \
                      H2["cccc"][0,l,k,i,j] = H2["cccc"][0,k,l,j,i] = val
                  H2["cccc"][0,j,i,l,k] = H2["cccc"][0,i,j,k,l] = \
                      H2["cccc"][0,l,k,j,i] = H2["cccc"][0,k,l,i,j] = -val
        else: # bogoliubov, not restricted
            H0 = 0
            H1 = {
                "cd": np.zeros((2,norb, norb)),
                "cc": np.zeros((1,norb, norb))
            }
            H2 = {
                "ccdd": np.zeros((3,norb, norb, norb, norb)),
                "cccd": np.zeros((2,norb, norb, norb, norb)),
                "cccc": np.zeros((1,norb, norb, norb, norb))
            }
            lines = f.readlines()
            section = 0
            for line in lines:
                tokens = line.split()
                val = float(tokens[0])
                i,j,k,l = [int(x) - 1 for x in tokens[1:]]
                if i < 0 and j < 0 and k < 0 and l < 0:
                    section += 1
                    H0 += val
                if section == 0 or section == 1:
                    H2["ccdd"][section, i,j,k,l] = H2["ccdd"][section,k,l,i,j] = val
                elif section == 2:
                    H2["ccdd"][2,i,j,k,l] = val
                elif section == 3 or section == 4: # cccdA/cccdB
                    H2["cccd"][section-3,i,j,k,l] = val
                    H2["cccd"][section-3,j,i,k,l] = -val
                elif section == 5:
                    H2["cccc"][0,i,j,l,k] = H2["cccc"][0,j,i,k,l] = val
                    H2["cccc"][0,j,i,l,k] = H2["cccc"][0,i,j,k,l] = -val
                elif section == 6 or section == 7:
                    log.eassert(k == -1 and l == -1, "Integral Syntax unrecognized")
                    H1["cd"][section-6,i,j] = H1["cd"][section-6,j,i] = val
                elif section == 8:
                    log.eassert(k == -1 and l == -1, "Integral Syntax unrecognized")
                    H1["cc"][0,i,j] = val
    return Integral(norb, restricted, bogoliubov, H0, H1, H2)

def dumpHDF5(filename, integral):
    log.error("function not implemented: dump_bin")
    raise Exception

def readHDF5(filename, norb, restricted, bogoliubov):
    import h5py
    log.eassert(h5py.is_hdf5(filename), "File %s is not hdf5 file", filename)
    f = h5py.File(filename)
    log.eassert(f["restricted"] == restricted, "spin restriction is not consistent")
    log.eassert(f["bogoliubov"] == bogoliubov, "particle number conservation is not consistent")
    log.eassert(f["norb"] == norb, "orbital number is not consistent")
    log.error("function not implemented: read_bin")
    raise Exception

def dumpMMAP(filename, integral):
    log.eassert(os.path.isdir(filename), "unable to dump memory map files")

    def mmap_write(itype, data):
        temp = np.memmap(os.path.join(filename, itype + ".mmap"), dtype="float", mode='w+', shape=data.shape)
        temp[:] = data[:]
        del temp

    for key, data in integral.H1.items():
        mmap_write(key, data)
    for key, data in integral.H2.items():
        mmap_write(key, data)

    temp = np.memmap(os.path.join(filename, "H0.mmap"), dtype="float", mode='w+', shape=(1,))
    temp[0] = integral.H0
    del temp

def readMMAP(filename, norb, restricted, bogoliubov, copy=False):
    log.eassert(os.path.isdir(filename), "unable to read memory map files")

    def bind(itype, shape):
        if copy:
            return np.array(np.memmap(os.path.join(filename, itype+".mmap"), dtype="float", mode='r', shape=shape))
        else:
            return np.memmap(os.path.join(filename, itype + ".mmap"), dtype="float", mode='r+', shape=shape)

    if restricted and not bogoliubov:
        H1 = {
            "cd": bind("cd", (1,norb, norb))
        }
        H2 = {
            "ccdd": bind("ccdd", (1,norb, norb, norb, norb))
        }
    elif not restricted and not bogoliubov:
        H1 = {
            "cd": bind("cd", (2,norb, norb))
        }
        H2 = {
            "ccdd": bind("ccdd", (3,norb, norb, norb, norb)),
        }
    elif restricted and bogoliubov:
        H1 = {
            "cd": bind("cd", (1,norb, norb)),
            "cc": bind("cc", (1,norb, norb))
        }
        H2 = {
            "ccdd": bind("ccdd", (1,norb, norb, norb, norb)),
            "cccd": bind("cccd", (1,norb, norb, norb, norb)),
            "cccc": bind("cccc", (1,norb, norb, norb, norb)),
        }
    else:
        H1 = {
            "cd": bind("cd", (2,norb, norb)),
            "cc": bind("cc", (1,norb, norb)),
        }
        H2 = {
            "ccdd": bind("ccdd", (3,norb, norb, norb, norb)),
            "cccd": bind("cccd", (2,norb, norb, norb, norb)),
            "cccc": bind("cccc", (1,norb, norb, norb, norb)),
        }
    H0 = bind("H0", (1,))[0]

    return Integral(norb, restricted, bogoliubov, H0, H1, H2)

def read(filename, norb, restricted, bogoliubov, fmt, **kwargs):
    if fmt == "FCIDUMP":
        return readFCIDUMP(filename, norb, restricted, bogoliubov, **kwargs)
    elif fmt == "HDF5":
        return readHDF5(filename, norb, restricted, bogoliubov, **kwargs)
    elif fmt == "MMAP":
        return readMMAP(filename, norb, restricted, bogoliubov, **kwargs)
    else:
        raise Exception("Unrecognized formt %s" % fmt)

def dump(filename, Ham, fmt, **kwargs):
    if fmt == "FCIDUMP":
        return dumpFCIDUMP(filename, Ham, **kwargs)
    elif fmt == "HDF5":
        return dumpHDF5(filename, Ham, **kwargs)
    elif fmt == "MMAP":
        return dumpMMAP(filename, Ham, **kwargs)
    else:
        raise Exception("Unrecognized formt %s" % fmt)

def get_eri_format(eri, nao):
    """
    Get the format of ERI, which can be the following:
        1. spin_dim: 0, 1, 3
        2. s8, s4, s1
    """
    eri = np.asarray(eri)
    nao_pair = nao * (nao + 1) // 2
    nao_pair_pair = (nao_pair) * (nao_pair + 1) // 2
    s1_size = nao ** 4
    s4_size = nao_pair * nao_pair
    s8_size = nao_pair_pair
    
    if eri.ndim == 5: 
        # s1 with spin = 1 or 3
        eri_format = 's1'
        spin_dim = eri.size // s1_size
        assert spin_dim * s1_size == eri.size
    elif (eri.ndim == 4) and (eri.size == s1_size): 
        # s1 with spin = 0
        eri_format = 's1'
        spin_dim = 0
    elif eri.ndim == 3:
        # s4 with spin = 1 or 3
        eri_format = 's4'
        spin_dim = eri.size // s4_size
        assert spin_dim * s4_size == eri.size
    elif eri.ndim == 2 and (eri.size == s4_size):
        # s4 with spin = 1 or 3
        eri_format = 's4'
        spin_dim = 0
    elif eri.ndim == 2 and (eri.size == s8_size):
        # s8 with spin = 1
        eri_format = 's8'
        spin_dim = 1
    elif eri.ndim == 1 and (eri.size == s8_size):
        eri_format = 's8'
        spin_dim = 0
    else:
        raise ValueError("Unknown ERI shape %s" %(str(eri.shape)))
    assert spin_dim in [0, 1, 3] 
    return eri_format, spin_dim

def check_perm_symm(eri, tol=1e-8):
    """
    Check the permutation symmetry of a plain eri in chemists' notation.
    
    Conventions:
    Sherrill's notes (4-fold symmetry): 
    real: (ij|kl) = (ji|lk)
    spin (for aaaa and bbbb type integral): (ij|kl) = (kl|ij)
    Combining these two, we have: (ij|kl) = (lk|ji) [hermi]

    PySCF's convention on 4-fold symmetry:
    permute over the first two: (ij|kl) = (ji|kl)
    permute over the last two: (ij|kl) = (ij|lk)
    Combining these two, we have: (ij|kl) = (ji|lk) [real]
    Note that PySCF's 4-fold symmetrized ERI always 
    has a shape of (nao_pair, nao_pair).
    If [spin] symmetry is further considered, it is 8-fold.
    
    Args:
        eri: H2, shape (nao, nao, nao, nao), real
        tol: tolerance for symmetry
    """
    log.info("Check permutation symmetry of eri.")
    eri = np.asarray(eri)
    if eri.ndim == 4:
        if np.iscomplexobj(eri):
            log.warn("eri is complex.")
        # check pyscf symmerty
        log.info("pyscf's symmetry")
        ij_perm = misc.max_abs(eri - eri.transpose(1, 0, 2, 3)) < tol
        log.info("ij symm:    (ij|kl) == (ji|kl)  ? %s", ij_perm)
        
        kl_perm = misc.max_abs(eri - eri.transpose(0, 1, 3, 2)) < tol
        log.info("kl symm:    (ij|kl) == (ij|lk)  ? %s", kl_perm)
        
        spin_perm = misc.max_abs(eri - eri.transpose(2, 3, 0, 1)) < tol
        log.info("spin symm:  (ij|kl) == (kl|ij)  ? %s", spin_perm)
        
        # check Sherrill's symmetry
        log.info("\nSherill's symmetry")
        real_perm = misc.max_abs(eri - eri.transpose(1, 0, 3, 2)) < tol
        log.info("real symm:  (ij|kl) == (ji|lk)  ? %s", real_perm)
        log.info("spin symm:  (ij|kl) == (kl|ij)  ? %s", spin_perm)
        
        hermi_perm = misc.max_abs(eri - eri.transpose(3, 2, 1, 0).conj()) < tol
        log.info("hermi symm: (ij|kl) == (lk|ji)* ? %s \n", hermi_perm)
    elif eri.ndim == 2:
        # check 8-fold symmetry
        spin_perm = misc.max_abs(eri - eri.T) < tol
        log.info("spin symm:  (ij|kl) == (kl|ij)  ? %s", spin_perm)
    else:
        raise ValueError("eri shape %s is not correct."%eri.shape)

if __name__ == "__main__":
    test()
