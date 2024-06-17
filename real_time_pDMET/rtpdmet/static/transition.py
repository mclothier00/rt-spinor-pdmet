import time
import numpy as np
import real_time_pDMET.rtpdmet.dynamics.system_mod as system_mod
import real_time_pDMET.rtpdmet.dynamics.system_mod_spinor as system_mod_spinor
import real_time_pDMET.rtpdmet.dynamics.fragment_mod as fragment_mod_dynamic
import real_time_pDMET.rtpdmet.dynamics.fragment_mod_spinor as fragment_mod_dynamic_spinor
import real_time_pDMET.rtpdmet.static.fragment_mod as fragment_mod
#import rt_dmet_toedit.rtpdmet.static.pDMET_glob as pdmet_glob
from pyscf import gto, scf, ao2mo, fci
import itertools

def transition(
        the_dmet, Nsites, Nele, Nfrag, impindx, h_site,
        V_site, hamtype, hubsite_indx, periodic):
    transition_time = time.time()

    mf1RDM = the_dmet.mf1RDM
    tot_system = system_mod.system(
        Nsites, Nele, Nfrag, impindx, h_site, V_site,
        hamtype, mf1RDM, hubsite_indx, periodic)
    tot_system.glob1RDM = the_dmet.glob1RDM
    tot_system.mf1RDM = the_dmet.mf1RDM
    tot_system.NOevecs = the_dmet.NOevecs
    tot_system.NOevals = the_dmet.NOevals
    tot_system.frag_list = []
    for i in range(Nfrag):
        tot_system.frag_list.append(
            fragment_mod_dynamic.fragment(impindx[i], Nsites, Nele))
        tot_system.frag_list[i].rotmat = the_dmet.frag_list[i].rotmat
        tot_system.frag_list[i].CIcoeffs = the_dmet.frag_list[i].CIcoeffs
    return tot_system


# NOTE: TEST THIS, many functions are new
def rtog_transition(
        the_dmet, Nsites, Nele, Nfrag, impindx, h_site_r,
        V_site_r, hamtype, hubsite_indx, periodic):

    transition_time = time.time()
    mf1RDM_r = the_dmet.mf1RDM

    # changing hamiltonian and 1RDM from restricted to generalized
    # NOTE: if new hamiltonian is created to introduce dynamics (as done in Dariia's examples),
    #       this is redundant. I'm leaving it in in case dynamics are initialized by something other
    #       than a change in the Hamiltonian
    h_site = np.kron(np.eye(2), h_site_r)
    V_site = block_tensor(V_site_r)

    mf1RDM = np.kron(np.eye(2), 0.5*mf1RDM_r)

    # changing impindx to reflect spinors in block diagonal form
    # ex: ([0, 1], [2, 3]) --> ([0, 1, 4, 5], [2, 3, 6, 7])
    
    # currently in spin block indexing
    impindx = spinor_impindx(Nsites, Nfrag)

    # NOTE: also possibly redundant, for same reason as h_site and V_site. implementing anyway
    hubsite_indx = spinor_hubsite(hubsite_indx, Nsites)

    tot_system = system_mod_spinor.system(
        Nsites, Nele, Nfrag, impindx, h_site, V_site,
        hamtype, mf1RDM, hubsite_indx, periodic)
    
    tot_system.glob1RDM = np.kron(np.eye(2), 0.5*the_dmet.glob1RDM)
    tot_system.mf1RDM = mf1RDM
    tot_system.NOevals, tot_system.NOevecs = get_nat_orbs(tot_system.glob1RDM)

    tot_system.frag_list = []
    nbeta = Nele//2
    nalpha = Nele - nbeta
    for i in range(Nfrag):
        frag_i = fragment_mod_dynamic_spinor.fragment(impindx[i], Nsites, Nele)
        tot_system.frag_list.append(frag_i)
        tot_system.frag_list[i].rotmat = frag_i.get_rotmat(mf1RDM)[0]
        tot_system.frag_list[i].CIcoeffs = to_gen_coeff(Nsites, Nsites, (Nsites*2), nalpha, nbeta, the_dmet.frag_list[i].CIcoeffs) 

    print('currently setting tot_sysem.mf1RDM (and tot_system.glob1RDM) to the reshaped mf1RDM (glob1RDM)... theres also the option of the intialize_GHF call for the mf1RDM and the get_glob1RDM for the glob1RDM')

    return tot_system

### NOTE: will need a utog_transition:
# def utog_transition...



def concat_strings(alphastr, betastr, norb_alpha, norb_beta):

    matrix_elements = []

    for i in range(len(alphastr)): 
        for j in range(len(betastr)):
    
            # convert binary numbers to strings and remove prefix
            alpha_str = bin(alphastr[i])[2:]
            beta_str = bin(betastr[j])[2:]
        
            # adds leading zeros so both strings are of the same length
            alpha_str = alpha_str.zfill(max(norb_alpha, norb_beta))
            beta_str = beta_str.zfill(max(norb_alpha, norb_beta))
        
            # concatenate strings
            matrix_str = "".join(i for j in zip(beta_str, alpha_str) for i in j)
            matrix_elements.append(int('0b' + matrix_str, 2))
    
    return matrix_elements


def to_gen_coeff(norb_alpha, norb_beta, norb_gen, nalpha, nbeta, coeffs):

    nelec = nalpha + nbeta

    # spinor indices
    strings = fci.cistring.make_strings(np.arange(norb_gen), nelec)

    # matrix row, column indices
    alphastr = fci.cistring.make_strings(np.arange(norb_alpha), nalpha)
    betastr = fci.cistring.make_strings(np.arange(norb_beta), nbeta)

    # matrix elements 

    matrix_elements = concat_strings(alphastr, betastr, norb_alpha, norb_beta)

    ### mapping from newly created strings to strings from make_strings
    new_coeff = np.zeros(len(strings), dtype=complex)
    coeffs = np.matrix.flatten(coeffs)

    for i in range(len(coeffs)):
        index = fci.cistring.str2addr(norb_gen, nelec, matrix_elements[i])
        new_coeff[index] = coeffs[i]

    return new_coeff


## copied over from pDMET_glob.py

def initialize_GHF(Nele, h_site, V_site):
    print("Mf 1RDM is initialized with GHF")
    Norbs = Nele
    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True
    mf = scf.GHF(mol)
    mf.get_hcore = lambda *args: h_site
    mf.get_ovlp = lambda *args: np.eye(Norbs)
    mf._eri = ao2mo.restore(8, V_site, Norbs)

    mf.kernel()
    mfRDM = mf.make_rdm1()

    return mfRDM


## NOTE: currently not sure if this correctly handles symmetries!! something we need
## to consider
def block_tensor(a):
    ## creates a "block diagonal" tensor from a given tensor

    shape1, shape2, shape3, shape4 = np.shape(a)
    ten_block = np.zeros((shape1*2, shape2*2, shape3*2, shape4*2))
    ten_block[:shape1, :shape2, :shape3, :shape4] = a[:, :, :, :]
    ten_block[shape1:, shape2:, shape3:, shape4:] = a[:, :, :, :]

    return ten_block


def spinor_impindx(Nsites, Nfrag, spinblock=True):
    ## creates a new impindx based on spinor (or unrestricted) orbitals

    impindx = []

    if spinblock == True:
        # spinor, aaaabbbb configuration
        Nimp = int(Nsites/Nfrag)
        for i in range(Nfrag):
            impindx.append(np.concatenate((np.arange(i*Nimp, (i+1)*Nimp), np.arange(i*Nimp+Nsites, (i+1)*Nimp+Nsites))))
    else:
        # spinor, abababab configuration
        gNimp = int((Nsites*2)/Nfrag)
        for i in range(Nfrag):
            impindx.append(np.arange(i*gNimp, (i+1)*gNimp))

    return impindx 


def spinor_hubsite(hubsite_indx, Nsites):
    ## converts a hubsite_indx to spinor (or unrestricted) indices 
    ## NOTE: currently only created for spin blocked indexing

    spinor_hubsite_indx = []
    
    for i in range(len(hubsite_indx)):
        spinor_hubsite_indx.append(hubsite_indx[i])
        spinor_hubsite_indx.append(hubsite_indx[i] + Nsites)
    
    spinor_hubsite_indx = np.array(spinor_hubsite_indx)

    return spinor_hubsite_indx


def get_nat_orbs(glob1RDM):
    # Subroutine to obtain natural orbitals of global 1RDM
    NOevals, NOevecs = np.linalg.eigh(glob1RDM)
    
    # Re-order such that eigenvalues are in descending order
    NOevals = np.flip(NOevals)
    NOevecs = np.flip(NOevecs, 1)
    
    return NOevals, NOevecs


### may not need these; spin staggered indices 

def reshape_rtog_matrix(a):
    ## reshape a block diagonal matrix a to a generalized form with 1a,1b,2a,2b, etc.

    num_rows, num_cols = a.shape
    block_indices = np.arange(num_cols)
    spin_block_size = int(num_cols/2)
    
    alpha_block = block_indices[:spin_block_size]
    beta_block = block_indices[spin_block_size:]
    
    indices = [list(itertools.chain(i))
                for i in zip(alpha_block, beta_block)]
    
    indices = np.asarray(indices).reshape(-1)

    new_a = a[:, indices]

    return new_a


def reshape_rtog_tensor(a):
    ## reshape a block diagonal tensor a to a generalized form with columns as 1a,1b,2a,2b, etc.

    num_rows, num_cols, dim1, dim2 = a.shape
    block_indices = np.arange(num_cols)
    spin_block_size = int(num_cols/2)
    
    alpha_block = block_indices[:spin_block_size]
    beta_block = block_indices[spin_block_size:]
    
    indices = [list(itertools.chain(i))
                for i in zip(alpha_block, beta_block)]

    indices = np.asarray(indices).reshape(-1)

    new_a = a[:, :, :, indices]

    return new_a



