# Mod that contatins subroutines necessary to calculate the analytical form
# of the first time-derivative of the mean-field 1RDM
# this is necessary when integrating the MF 1RDM and CI coefficients explicitly
# while diagonalizing the MF1RDM at each time-step to obtain embedding orbitals

import real_time_pDMET.scripts.utils as utils
import numpy as np
import multiprocessing as multproc

import time


#####################################################################

# NOTE: this doesn't seem to be used anywhere


def get_ddt_mf_NOs(system, G_site):
    ddt_mf1RDM = -1j * (np.dot(G_site, system.mf1RDM) - np.dot(system.mf1RDM, G_site))
    ddt_NOevecs = -1j * np.dot(G_site, system.NOevecs)

    return ddt_mf1RDM, ddt_NOevecs


#####################################################################


def get_ddt_glob(dG, system):
    system.get_frag_iddt_corr1RDM()

    # Calculate i times time-derivative of global 1RDM
    iddt_glob1RDM = calc_iddt_glob1RDM(system)

    # Calculate G-matrix governing time-dependence of natural orbitals
    # This G is in the natural orbital basis
    Gmat_time = time.time()
    G_site = calc_Gmat(dG, system, iddt_glob1RDM)
    ddt_glob1RDM = -1j * iddt_glob1RDM

    return ddt_glob1RDM, G_site


#####################################################################


def get_ddt_mf1rdm_serial(dG, system, Nocc):
    # Subroutine to solve for the time-dependence of the MF 1RDM
    # this returns the time-derivative NOT i times the time-derivative

    # NOTE: prior to this routine being called, necessary to have
    # the rotation matrices, 1RDM, and 2RDM for each fragment
    # as well as the natural orbitals and eigenvalues of the global
    # 1RDM previously calculated

    # Calculate the Hamiltonian commutator portion
    # of the time-dependence of correlated 1RDM for each fragment

    # ie i\tilde{ \dot{ correlated 1RDM } } using notation from notes
    system.get_frag_iddt_corr1RDM()

    # Calculate i times time-derivative of global 1RDM
    iddt_glob1RDM = calc_iddt_glob1RDM(system)

    # Calculate G-matrix governing time-dependence of natural orbitals
    # This G is in the natural orbital basis
    Gmat_time = time.time()
    G_site = calc_Gmat(dG, system, iddt_glob1RDM)

    ddt_mf1RDM = -1j * (np.dot(G_site, system.mf1RDM) - np.dot(system.mf1RDM, G_site))
    ddt_NOevecs = -1j * np.dot(G_site, system.NOevecs)

    ddt_glob1RDM = -1j * iddt_glob1RDM

    # Calculate alternative time-derivative of MF 1RDM
    # This method is more sensitive to numerical instabilitied associated with
    # NO degeneracies
    if not system.gen:
        short_NOcc = np.copy(system.NOevecs[:, : round(system.Nele / 2)])
        short_ddtNOcc = np.copy(ddt_NOevecs[:, : round(system.Nele / 2)])
        chk = 2 * (
            np.dot(short_ddtNOcc, short_NOcc.conj().T)
            + np.dot(short_NOcc, short_ddtNOcc.conj().T)
        )

    # NOTE: PING is calculation of short_NOcc still the same??
    if system.gen:
        short_NOcc = np.copy(system.NOevecs[:, : round(system.Nele / 2)])
        short_ddtNOcc = np.copy(ddt_NOevecs[:, : round(system.Nele / 2)])
        chk = np.dot(short_ddtNOcc, short_NOcc.conj().T) + np.dot(
            short_NOcc, short_ddtNOcc.conj().T
        )

    ddtmf1RDM_check = np.allclose(chk, ddt_mf1RDM, rtol=0, atol=1e-5)

    f = open("output_halffrag.txt", "a")
    f.write("\n TD of global density matrix \n")
    f.close()
    utils.printarray(ddt_glob1RDM, "output_halffrag.txt")
    f = open("output_halffrag.txt", "a")
    f.write("\n G \n")
    f.close()
    utils.printarray(G_site, "output_halffrag.txt")
    f = open("output_halffrag.txt", "a")
    f.write("\n TD of NO evals (U dot) \n")
    f.close()
    utils.printarray(ddt_NOevecs, "output_halffrag.txt")
    f = open("output_halffrag.txt", "a")
    f.write("\n TD of mean field density matrix \n")
    f.close()
    utils.printarray(ddt_mf1RDM, "output_halffrag.txt")

    return ddt_glob1RDM, ddt_NOevecs, ddt_mf1RDM, G_site, ddtmf1RDM_check


#####################################################################

# currently editing, but not yet checking


def calc_iddt_glob1RDM(system):
    # Subroutine to calculate i times
    # time dependence of global 1RDM forcing anti-hermiticity
    if not system.gen:
        rotmat_unpck = np.zeros(
            [system.Nsites, system.Nsites, system.Nsites], dtype=complex
        )
        iddt_corr1RDM_unpck = np.zeros([system.Nsites, system.Nsites], dtype=complex)

        for q in range(system.Nsites):
            # fragment for site q
            frag = system.frag_list[system.site_to_frag_list[q]]

            # index within fragment corresponding to site q -
            # note that q is an impurity orbital
            qimp = system.site_to_impindx[q]

            # unpack rotation matrix
            rotmat_unpck[:, :, q] = np.copy(frag.rotmat)

            # unpack necessary portion of iddt_corr1RDM
            iddt_corr1RDM_unpck[:, q] = np.copy(frag.iddt_corr1RDM[:, qimp])

    if system.gen:
        rotmat_unpck = np.zeros(
            [2 * system.Nsites, 2 * system.Nsites, 2 * system.Nsites], dtype=complex
        )
        iddt_corr1RDM_unpck = np.zeros(
            [2 * system.Nsites, 2 * system.Nsites], dtype=complex
        )

        for q in range(2 * system.Nsites):
            # fragment for site q
            frag = system.frag_list[system.site_to_frag_list[q]]

            # index within fragment corresponding to site q -
            # note that q is an impurity orbital
            qimp = system.site_to_impindx[q]

            # unpack rotation matrix
            rotmat_unpck[:, :, q] = np.copy(frag.rotmat)

            # unpack necessary portion of iddt_corr1RDM
            iddt_corr1RDM_unpck[:, q] = np.copy(frag.iddt_corr1RDM[:, qimp])

    # calculate intermediate matrix
    tmp = np.einsum("paq,aq->pq", rotmat_unpck, iddt_corr1RDM_unpck)

    return 0.5 * (tmp - tmp.conj().T)


#####################################################################


# currently editing; not yet checking...
# NOTE: only edit was expansion of 2 * system.Nsites


def calc_Gmat(dG, system, iddt_glob1RDM):
    # Subroutine to calculate matrix that
    # governs time-dependence of natural orbitals

    # Matrix of one over the difference in global 1RDM eigenvalues
    # Set diagonal terms and terms where eigenvalues are almost equal to zero
    evals = np.copy(system.NOevals)
    # G2_fast_time = time.time()
    G2_fast = utils.rot1el(iddt_glob1RDM, system.NOevecs)

    if not system.gen:
        for a in range(system.Nsites):
            for b in range(system.Nsites):
                if a != b and np.abs(evals[a] - evals[b]) > dG:
                    G2_fast[a, b] /= evals[b] - evals[a]
                else:
                    G2_fast[a, b] = 0

    if system.gen:
        for a in range(2 * system.Nsites):
            for b in range(2 * system.Nsites):
                if a != b and np.abs(evals[a] - evals[b]) > dG:
                    G2_fast[a, b] /= evals[b] - evals[a]
                else:
                    G2_fast[a, b] = 0

    G2_fast = np.triu(G2_fast) + np.triu(G2_fast, 1).conjugate().transpose()
    G2_site = utils.rot1el(G2_fast, utils.adjoint(system.NOevecs))

    return G2_site


#####################################################################
