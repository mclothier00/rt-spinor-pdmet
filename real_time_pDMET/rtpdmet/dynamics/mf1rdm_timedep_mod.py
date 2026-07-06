# Mod that contatins subroutines necessary to calculate the analytical form
# of the first time-derivative of the mean-field 1RDM
# this is necessary when integrating the MF 1RDM and CI coefficients explicitly
# while diagonalizing the MF1RDM at each time-step to obtain embedding orbitals

import real_time_pDMET.scripts.utils as utils
import numpy as np
import multiprocessing as multproc
from mpi4py import MPI

import time


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


def get_ddt_mf1rdm_serial(dG, system):
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

    if system.gen:
        short_NOcc = np.copy(system.NOevecs[:, : round(system.Nele)])
        short_ddtNOcc = np.copy(ddt_NOevecs[:, : round(system.Nele)])
        chk = np.dot(short_ddtNOcc, short_NOcc.conj().T) + np.dot(
            short_NOcc, short_ddtNOcc.conj().T
        )

    ddtmf1RDM_check = np.allclose(chk, ddt_mf1RDM, rtol=0, atol=1e-9)

    return ddt_glob1RDM, ddt_NOevecs, ddt_mf1RDM, G_site, ddtmf1RDM_check


#####################################################################


def calc_iddt_glob1RDM(system):
    # Subroutine to calculate i times
    # time dependence of global 1RDM forcing anti-hermiticity
    Nsites = system.Nsites

    iddt_glob1RDM = np.zeros([Nsites, Nsites], dtype=complex)

    for i, frag in enumerate(system.frag_in_rank):
        tmp = 0.5 * np.dot(
            frag.rotmat, np.dot(frag.iddt_corr1RDM, frag.rotmat.conj().T)
        )
        for site in frag.impindx:
            iddt_glob1RDM[site, :] += tmp[site, :]
            iddt_glob1RDM[:, site] += tmp[:, site]

    true_iddt_glob1RDM = np.zeros([Nsites, Nsites], dtype=complex)
    MPI.COMM_WORLD.Allreduce(iddt_glob1RDM, true_iddt_glob1RDM, op=MPI.SUM)

    return true_iddt_glob1RDM


#####################################################################


# def calc_Gmat(dG, system, iddt_glob1RDM):
#     # Subroutine to calculate matrix that
#     # governs time-dependence of natural orbitals
#
#     # Matrix of one over the difference in global 1RDM eigenvalues
#     # Uses Tikhonov regularization
#
#     evals = np.copy(system.NOevals)
#     G2_fast = utils.rot1el(iddt_glob1RDM, system.NOevecs)
#
#     for a in range(system.Nsites):
#         for b in range(system.Nsites):
#             if a != b:
#                 d = evals[b] - evals[a]
#                 G2_fast[a, b] *= d / (d**2 + dG**2)
#
#     G2_fast = 0.5 * (G2_fast + G2_fast.conj().T)
#
#     if not np.allclose(G2_fast, G2_fast.conj().T, atol=1e-10):
#         print(
#             f"WARNING: Gmat not Hermitian: {np.max(np.abs(G2_fast + G2_fast.conj().T))}"
#         )
#
#     G2_site = utils.rot1el(G2_fast, utils.adjoint(system.NOevecs))
#
#     return G2_site


def calc_Gmat(dG, system, iddt_glob1RDM):
    # Subroutine to calculate matrix that
    # governs time-dependence of natural orbitals

    # Matrix of one over the difference in global 1RDM eigenvalues,
    # smoothed by a Tikhonov regularizer with width dG.
    # Near degenerate eigenvalues are handled continuously; see get_Xmat.

    evals = np.copy(system.NOevals)
    G2_fast = utils.rot1el(iddt_glob1RDM, system.NOevecs)

    for a in range(system.Nsites):
        for b in range(system.Nsites):
            d = evals[b] - evals[a]
            G2_fast[a, b] *= d / (d**2 + dG**2)

    G2_fast = 0.5 * (G2_fast + G2_fast.conj().T)

    if not np.allclose(G2_fast, G2_fast.conj().T, atol=1e-10):
        print(
            f"WARNING: Gmat not Hermitian: {np.max(np.abs(G2_fast + G2_fast.conj().T))}"
        )

    G2_site = utils.rot1el(G2_fast, utils.adjoint(system.NOevecs))

    return G2_site


#####################################################################
