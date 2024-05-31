import time
import numpy as np
import real_time_pDMET.rtpdmet.dynamics.system_mod as system_mod
import real_time_pDMET.rtpdmet.dynamics.fragment_mod as fragment_mod_dynamic
import real_time_pDMET.rtpdmet.static.fragment_mod as fragment_mod
#import rt_dmet_toedit.rtpdmet.static.pDMET_glob as pdmet_glob
from pyscf import gto, scf, ao2mo
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


def rtog_transition(
        the_dmet, Nsites, Nele, Nfrag, impindx, h_site_r,
        V_site_r, hamtype, hubsite_indx, periodic):
    transition_time = time.time()
    mf1RDM_r = the_dmet.mf1RDM

    #changing hamiltonian and 1RDM from restricted to generalized
    h_site = np.kron(np.eye(2), h_site_r)
    V_site = np.kron(np.eye(2), V_site_r)

    print(h_site)

    # NOTE: CHECK THIS, what is this doing?
    mf1RDM_r = 0.5*mf1RDM_r
    mf1RDM = np.kron(np.eye(2), mf1RDM_r)

    # NOTE: TEST THIS NEW CODE
    h_site = reshape_rtog(h_site)
    print(h_site)
    #DOES NOT WORK FOR A TENSOR
    V_site = reshape_rtog(V_site)
    mf1RDM = reshape_rtog(mf1RDM)


    tot_system = system_mod.system(
        Nsites, Nele, Nfrag, impindx, h_site, V_site,
        hamtype, mf1RDM, hubsite_indx, periodic)
    
    tot_system.frag_list = []
    for i in range(Nfrag):
        tot_system.frag_list.append(
            fragment_mod_dynamic.fragment(impindx[i], Nsites, Nele))
        tot_system.frag_list[i].rotmat = the_dmet.frag_list[i].rotmat # not a big deal
        tot_system.frag_list[i].CIcoeffs = the_dmet.frag_list[i].CIcoeffs # kinda a big deal, no way to regenerate
        # use CI coefficients to recreate corr1RDM

    # NOT NEEDED? will have to recreate them at beginning of dynamics code
    #glob1RDM = the_dmet.glob1RDM
    #tot_system.glob1RDM = get_globalRDM(Nsites, tot_system.frag_list, impindx)

    tot_system.mf1RDM = initialize_GHF(Nele, h_site, V_site)

    # WHAT TO DO HERE?
    tot_system.NOevecs = the_dmet.NOevecs
    tot_system.NOevals = the_dmet.NOevals

    return tot_system

### NOTE: need to get transition function to work with what the to_gen_coeff definition requires

def concat_strings(alphastr, betastr):
    matrix_elements = []
    for i in range(len(alphastr)): 
        for j in range(len(betastr)):
            # convert binary numbers to strings and remove prefix
            alpha_str = bin(alphastr[i])[2:]
            beta_str = bin(betastr[j])[2:]
        
            # adds leading zeros so both strings are of the same length
            alpha_str = alpha_str.zfill(3)
            beta_str = beta_str.zfill(3)
        
            # concatenate strings
            matrix_str = "".join(i for j in zip(beta_str, alpha_str) for i in j)
            matrix_elements.append(int('0b' + matrix_str, 2))
    return matrix_elements


def to_gen_coeff(norb_alpha, norb_beta, norb_gen, nalpha, nbeta, coeffs):
    nelec = nalpha + nbeta

    # spinor indices
    strings = pyscf.fci.cistring.make_strings(np.arange(norb_gen), nelec)
    print(f'coefficient strings: {[bin(x) for x in strings]}')

    # matrix row, column indices
    alphastr = pyscf.fci.cistring.make_strings(np.arange(norb_alpha), nalpha)
    betastr = pyscf.fci.cistring.make_strings(np.arange(norb_beta), nbeta)
    print(f'{[bin(x) for x in alphastr]} \n {[bin(z) for z in betastr]}')

    # matrix elements 

    matrix_elements = concat_strings(alphastr, betastr)
    print(f'new coefficient strings from matrix: {[bin(x) for x in matrix_elements]}')

    ### mapping from newly created strings to strings from make_strings
    new_coeff = np.zeros(len(strings), dtype=complex)
    coeffs = np.matrix.flatten(coeffs)

    for i in range(len(coeffs)):
        index = pyscf.fci.cistring.str2addr(norb_gen, nelec, matrix_elements[i])
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




##### graveyard

def get_globalRDM(Nsites, frag_list, impindx):
    # initialize glodal 1RDM to be complex if rotation
    # matrix or correlated 1RDM is complex

    glob1RDM = np.zeros([2*Nsites, 2*Nsites])

    # copied from pDMET_glob.py
    site_to_frag_list = []
    site_to_impindx = []
    for i in range(Nsites):
        for ifrag, array in enumerate(impindx):
            if (i in array):
                site_to_frag_list.append(ifrag)
                site_to_impindx.append(np.argwhere(array == i)[0][0])

    # form the global 1RDM forcing hermiticity
    globalRDMtrace = 0
    for p in range(Nsites):
        for q in range(p, (Nsites)):

            # for both alpha and beta sites
            pfrag = frag_list[site_to_frag_list[p]]
            qfrag = frag_list[site_to_frag_list[q]]

        # index corresponding to the impurity and bath range
        # in the rotation matrix for each fragment
        # rotation matrix order:(sites) x (impurity, virtual, bath, core)

            pindx = np.r_[:pfrag.Nimp, pfrag.last_virt:pfrag.last_bath]
            qindx = np.r_[:qfrag.Nimp, qfrag.last_virt:qfrag.last_bath]

            glob1RDM[p, q] = 0.5 * np.linalg.multi_dot(
                [pfrag.rotmat[p, pindx], pfrag.corr1RDM,
                 pfrag.rotmat[q, pindx].conj().T])

            glob1RDM[p, q] += 0.5 * np.linalg.multi_dot(
                [qfrag.rotmat[p, qindx], qfrag.corr1RDM,
                 qfrag.rotmat[q, qindx].conj().T])

            if(p != q):  # forcing Hermiticity
                glob1RDM[q, p] = np.conjugate(glob1RDM[p, q])
    trace1RDM = glob1RDM.trace()
    print("trace of global RDM", trace1RDM)
    return glob1RDM

## copied over from fragment_mod.py
def get_rotmat(mf1RDM, impindx, Nsites, Nimp):

        '''
         Subroutine to generate rotation matrix from site to embedding basis
         PING currently impurities have to be listed in ascending order
         (though dont have to be sequential)
        '''

        # remove rows/columns corresponding to impurity sites from mf 1RDM
        # changed to check both spin blocks
        mf1RDM[:Nsites, :Nsites] = np.delete(mf1RDM, impindx, axis=0)
        mf1RDM[:Nsites, :Nsites] = np.delete(mf1RDM, impindx, axis=1)
        mf1RDM[Nsites:, Nsites:] = np.delete(mf1RDM, impindx, axis=0)
        mf1RDM[Nsites:, Nsites:] = np.delete(mf1RDM, impindx, axis=1)

        # diagonalize environment part of 1RDM to obtain embedding
        # (virtual, bath, core) orbitals
        evals, evecs = np.linalg.eigh(mf1RDM)

        # form rotation matrix consisting of unit vectors
        # for impurity and the evecs for embedding
        # rotation matrix is ordered as impurity, virtual, bath, core

        # WORKS ONLY FOR MULTI-IMPURITY INDEXING'''

        '''
        self.rotmat = np.zeros( [ self.Nsites, self.Nimp ] )
        for imp in range(self.Nimp):
            indx                     = self.impindx[imp]
            self.rotmat[ indx, imp ] = 1.0
            if indx <= evecs.shape[0]:
                evecs = np.insert( evecs, indx, 0.0, axis=0 )
            else:
                zero_coln = np.array([np.zeros(evecs.shape[1])])
                evecs = np.concatenate((evecs, zero_coln), axis=0)

        self.rotmat = np.concatenate( (self.rotmat,evecs), axis=1 )
        '''

        # WORKS FOR SINGLE IMPURITY INDEXING

        rotmat = np.zeros([(2*Nsites), (2*Nimp)])
        rmat = np.zeros([Nsites, Nimp])
        for imp in range(Nimp):
            indx = impindx[imp]
            rmat[indx, imp] = 1.0
            rotmat[:Nsites, :Nimp] = rmat 
            rotmat[Nsites:, Nimp:] = rmat
        # what's going on here? currently leaving alone
        if impindx[0] > impindx[Nimp-1]:
            for imp in range(Nimp):
                rev_impindx = np.flipud(impindx)
                indx = rev_impindx[imp]
                if indx <= evecs.shape[0]:
                    evecs = np.insert(evecs, indx, 0.0, axis=0)
                else:
                    print("index is  out of range, attaching zeros in the end")
                    zero_coln = np.array([np.zeros(evecs.shape[1])])
                    evecs = np.concatenate((evecs, zero_coln), axis=0)
        else:
            for imp in range(Nimp):
                indx = impindx[imp]
                if indx <= evecs.shape[0]:
                    evecs = np.insert(evecs, indx, 0.0, axis=0)
                else:
                    print("index is  ut of range, attaching zeros in the end")
                    zero_coln = np.array([np.zeros(evecs.shape[1])])
                    evecs = np.concatenate((evecs, zero_coln), axis=0)
        # worried about dimensions of evecs compared to rotmat, so printing
        print(F'printing eigenvectors: {evecs}')
        rotmat = np.concatenate((rotmat, evecs), axis=1)
        env1RDM_evals = evals
        return rotmat
    #####################################################################
