import time
import numpy as np
import real_time_pDMET.rtpdmet.dynamics.system_mod as system_mod
import real_time_pDMET.rtpdmet.dynamics.fragment_mod as fragment_mod_dynamic
import real_time_pDMET.scripts.utils as utils

# import rt_dmet_toedit.rtpdmet.static.pDMET_glob as pdmet_glob
from pyscf import gto, scf, ao2mo, fci
import itertools
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod


def rtor_transition(
    the_dmet,
    Nsites,
    Nele,
    Nfrag,
    impindx,
    h_site,
    V_site,
    hamtype,
    hubsite_indx,
    periodic,
):
    print(
        "Transitioning from restricted static calculation to restricted dynamic calculation."
    )

    transition_time = time.time()

    mf1RDM = the_dmet.mf1RDM
    tot_system = system_mod.system(
        Nsites,
        Nele,
        Nfrag,
        impindx,
        h_site,
        V_site,
        hamtype,
        mf1RDM,
        hubsite_indx,
        periodic,
    )
    tot_system.glob1RDM = the_dmet.glob1RDM
    tot_system.mf1RDM = the_dmet.mf1RDM
    tot_system.NOevecs = the_dmet.NOevecs
    tot_system.NOevals = the_dmet.NOevals
    tot_system.frag_list = []
    tot_system.Nbasis = Nsites  # for Hubbard
    for i in range(Nfrag):
        tot_system.frag_list.append(
            fragment_mod_dynamic.fragment(impindx[i], Nsites, Nele)
        )
        tot_system.frag_list[i].rotmat = the_dmet.frag_list[i].rotmat
        tot_system.frag_list[i].CIcoeffs = the_dmet.frag_list[i].CIcoeffs

    return tot_system


# NOTE: TEST THIS, many functions are new
def rtog_transition(
    the_dmet,
    Nsites,
    Nele,
    Nfrag,
    impindx,
    h_site_r,
    V_site_r,
    hamtype,
    hubsite_indx,
    periodic,
):
    print(
        "Transitioning from restricted static calculation to generalized dynamic calculation."
    )

    transition_time = time.time()

    # changing hamiltonian and 1RDM from restricted to generalized
    h_site = np.kron(np.eye(2), h_site_r)
    V_site = utils.block_tensor(V_site_r)
    h_site = utils.reshape_rtog_matrix(h_site)
    V_site = utils.reshape_rtog_tensor(V_site)

    # changing impindx to reflect spinors indexed via sites [ababab...]
    # ex: sites: ([0, 1], [2, 3]) --> ([0, 1, 2, 3], [4, 5, 6, 7])
    impindx = spinor_impindx(Nsites, Nfrag)

    hubsite_indx = spinor_hubsite(hubsite_indx, Nsites)

    mf1RDM = the_dmet.mf1RDM
    tot_system = system_mod.system(
        Nsites,
        Nele,
        Nfrag,
        impindx,
        h_site,
        V_site,
        hamtype,
        mf1RDM,
        hubsite_indx,
        periodic,
        gen=True,
    )

    tot_system.Nbasis = 2 * Nsites
    tot_system.glob1RDM = utils.reshape_rtog_matrix(
        np.kron(np.eye(2), 0.5 * the_dmet.glob1RDM)
    )
    tot_system.mf1RDM = utils.reshape_rtog_matrix(
        np.kron(np.eye(2), 0.5 * the_dmet.mf1RDM)
    )

    tot_system.NOevecs = utils.reshape_rtog_matrix(np.kron(np.eye(2), the_dmet.NOevecs))

    tot_system.NOevals = np.diag(
        np.dot(
            tot_system.NOevecs.conjugate().transpose(),
            np.dot(tot_system.glob1RDM, tot_system.NOevecs),
        )
    )

    tot_system.frag_list = []
    for i in range(Nfrag):
        frag_i = fragment_mod_dynamic.fragment(impindx[i], Nsites, Nele, gen=True)
        tot_system.frag_list.append(frag_i)
        tot_system.frag_list[i].rotmat = utils.reshape_rtog_matrix(
            np.kron(np.eye(2), the_dmet.frag_list[i].rotmat)
        )

        nbeta = frag_i.Nimp // 2
        nalpha = frag_i.Nimp - nbeta

        tot_system.frag_list[i].CIcoeffs = to_gen_coeff(
            frag_i.Nimp,
            frag_i.Nimp,
            (frag_i.Nimp * 2),
            nalpha,
            nbeta,
            the_dmet.frag_list[i].CIcoeffs,
        )

    print(
        "currently setting tot_sysem.mf1RDM (and tot_system.glob1RDM) to the reshaped mf1RDM (glob1RDM)... theres also the option of the intialize_GHF call for the mf1RDM and the get_glob1RDM for the glob1RDM"
    )

    # for hard-coded checks; have to manually adjust number of fragment arrays
    glob1rdm = tot_system.glob1RDM
    mf1rdm = tot_system.mf1RDM
    NOevecs = tot_system.NOevecs
    # currently only saves for two fragments
    rotmat_0 = tot_system.frag_list[0].rotmat
    rotmat_1 = tot_system.frag_list[1].rotmat
    CIcoeff_0 = tot_system.frag_list[0].CIcoeffs
    CIcoeff_1 = tot_system.frag_list[1].CIcoeffs

    np.savez(
        "matrices.npz",
        glob1rdm=glob1rdm,
        mf1rdm=mf1rdm,
        NOevecs=NOevecs,
        rotmat_0=rotmat_0,
        rotmat_1=rotmat_1,
        CIcoeff_0=CIcoeff_0,
        CIcoeff_1=CIcoeff_1,
    )

    return tot_system


def concat_strings(alphastr, betastr, norb_alpha, norb_beta):
    matrix_elements = []

    for i in range(len(alphastr)):
        for j in range(len(betastr)):
            det_info = []

            # convert binary numbers to strings and remove prefix
            alpha_str = bin(alphastr[i])[2:]
            beta_str = bin(betastr[j])[2:]

            # adds leading zeros so both strings are of the same length
            alpha_str = alpha_str.zfill(max(norb_alpha, norb_beta))
            beta_str = beta_str.zfill(max(norb_alpha, norb_beta))

            # concatenate strings
            matrix_str = "".join(i for j in zip(beta_str, alpha_str) for i in j)
            det_info.append(int("0b" + matrix_str, 2))

            # add alpha and beta strings to list matrix_elements
            det_info.append(alphastr[i])
            det_info.append(betastr[j])

            matrix_elements.append(det_info)

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
        index = fci.cistring.str2addr(norb_gen, nelec, matrix_elements[i][0])
        new_coeff[index] = coeffs[i]
        new_coeff[index] = coeffs[i] * coeff_parity_change(
            matrix_elements[i][1], matrix_elements[i][2], norb_alpha, norb_beta
        )

    return new_coeff


def coeff_parity_change(alphastr, betastr, nalpha, nbeta):
    # convert binary numbers to strings and remove prefix
    alpha_str = bin(alphastr)[2:]
    beta_str = bin(betastr)[2:]

    # adds leading zeros so both strings are of the same length
    alpha_str = alpha_str.zfill(max(nalpha, nbeta))
    beta_str = beta_str.zfill(max(nalpha, nbeta))

    # combines alpha and beta string into one restricted string
    res_str = beta_str + alpha_str
    resstr = int(res_str, 2)

    # if element in beta string is 1, determines parity
    new_parity = 1

    # NOTE: is this alphastr or rest of full string?
    for i, bit in enumerate(beta_str[::-1]):
        if bit == "1":
            # print(
            #    f"whats actually getting counted for {bin(resstr)} for {i}: {bin(alphastr >> (i+1))}"
            # )
            parity = (-1) ** bin(alphastr >> (i + 1)).count("1")
            new_parity = new_parity * parity
    # print(f"parity of {res_str}: {new_parity}")
    return new_parity


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


def spinor_impindx(Nsites, Nfrag, spinblock=False):
    ## creates a new impindx based on spinor (or unrestricted) orbitals

    impindx = []

    if spinblock:
        # spinor, aaaabbbb configuration
        Nimp = int(Nsites / Nfrag)
        for i in range(Nfrag):
            impindx.append(
                np.concatenate(
                    (
                        np.arange(i * Nimp, (i + 1) * Nimp),
                        np.arange(i * Nimp + Nsites, (i + 1) * Nimp + Nsites),
                    )
                )
            )
    else:
        # spinor, abababab configuration
        gNimp = int((Nsites * 2) / Nfrag)
        for i in range(Nfrag):
            impindx.append(np.arange(i * gNimp, (i + 1) * gNimp))

    return impindx


def spinor_hubsite(hubsite_indx, Nsites):
    ## converts a hubsite_indx to spinor (or unrestricted) indices

    spinor_hubsite_indx = []

    for i in range(len(hubsite_indx)):
        spinor_hubsite_indx.append(hubsite_indx[i] * 2)
        spinor_hubsite_indx.append(hubsite_indx[i] * 2 + 1)

    spinor_hubsite_indx = np.asarray(spinor_hubsite_indx)

    return spinor_hubsite_indx


def spinor_hubsite_block(hubsite_indx, Nsites):
    ## converts a hubsite_indx to spinor (or unrestricted) indices
    ## when in spin block form

    spinor_hubsite_indx = []

    for i in range(len(hubsite_indx)):
        spinor_hubsite_indx.append(hubsite_indx[i])
        spinor_hubsite_indx.append(hubsite_indx[i] + Nsites)

    spinor_hubsite_indx = np.asarray(spinor_hubsite_indx)

    return spinor_hubsite_indx


# NOTE: CHECK THIS


def get_nat_orbs(glob1RDM):
    # Subroutine to obtain natural orbitals of global 1RDM
    NOevals, NOevecs = np.linalg.eigh(glob1RDM)
    # Re-order such that eigenvalues are in descending order
    NOevals = np.flip(NOevals)
    NOevecs = np.flip(NOevecs, 1)
    return NOevals, NOevecs
