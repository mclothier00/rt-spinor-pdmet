import numpy as np
import real_time_pDMET.scripts.utils as utils
import pyscf.fci
import real_time_pDMET.scripts.applyham_pyscf as applyham_pyscf

# four spatial sites, 8 spinors, 4 electrons, two fragments
# following contains ['glob1rdm', 'mf1rdm', 'NOevecs', 'rotmat_0', 'rotmat_1', 'CIcoeff_0', 'CIcoeff_1', 'h1e', h2e']
output = np.load("matrices.npz")
init_glob = output["glob1rdm"].astype(complex)
init_mf = output["mf1rdm"].astype(complex)
init_noevec = output["NOevecs"].astype(complex)
init_rot0 = output["rotmat_0"].astype(complex)
init_rot1 = output["rotmat_1"].astype(complex)
init_ci0 = output["CIcoeff_0"].astype(complex)
init_ci1 = output["CIcoeff_1"].astype(complex)

ham = np.load("ham.npz")
h1e = ham["h1e"]
h2e = ham["h2e"]
h1e = np.kron(np.eye(2, dtype=int), h1e)
h2e = utils.block_tensor(h2e)
h1e = utils.reshape_rtog_matrix(h1e)
h2e = utils.reshape_rtog_tensor(h2e)
init_h1e = h1e.astype(complex)
init_h2e = h2e.astype(complex)
long_fmt = False


timestep = 0.001
steps = 10


def one_rk_step(delt, glob, mf, noevec, rot0, rot1, ci0, ci1, h1e, h2e):
    ##### checking getting new rotmat #####

    rot0 = np.zeros([8, 8], dtype=complex)
    rot1 = np.zeros([8, 8], dtype=complex)
    rot0[:4, :4] = np.identity(4, dtype=complex)
    rot1[4:, :4] = np.identity(4, dtype=complex)
    td_mf0 = mf[4:, 4:]
    td_mf1 = mf[:4, :4]
    mfevals0 = np.zeros([8], dtype=complex)
    mfevals1 = np.zeros([8], dtype=complex)
    mfevals0[:4], rot0[4:, 4:] = np.linalg.eigh(td_mf0)
    mfevals1[:4], rot1[:4, 4:] = np.linalg.eigh(td_mf1)

    ##### testing get_Hemb #####

    # rotating 1e terms for fragment 1
    h1emb0 = []
    h1emb1 = []
    rot0_adj = rot0.conjugate().transpose()
    rot1_adj = rot1.conjugate().transpose()
    h1emb0 = np.dot(rot0_adj, np.dot(h1e, rot0))
    h1emb1 = np.dot(rot1_adj, np.dot(h1e, rot1))

    # rotating 2e terms for fragment 1
    h2emb0 = np.zeros((8, 8, 8, 8), dtype=complex)
    h2emb1 = np.zeros((8, 8, 8, 8), dtype=complex)

    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in range(8):
                    sum0 = sum1 = 0.0
                    for z in range(8):
                        sum0 += h2e[i, j, k, z] * rot0[z, m]
                        sum1 += h2e[i, j, k, z] * rot1[z, m]
                    h2emb0[i, j, k, m] = sum0
                    h2emb1[i, j, k, m] = sum1

    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in range(8):
                    sum0 = sum1 = 0.0
                    for z in range(8):
                        sum0 += rot0_adj[k, z] * h2emb0[i, j, z, m]
                        sum1 += rot1_adj[k, z] * h2emb1[i, j, z, m]
                    h2emb0[i, j, k, m] = sum0
                    h2emb1[i, j, k, m] = sum1

    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in range(8):
                    sum0 = sum1 = 0.0
                    for z in range(8):
                        sum0 += h2emb0[i, z, k, m] * rot0[z, j]
                        sum1 += h2emb1[i, z, k, m] * rot1[z, j]
                    h2emb0[i, j, k, m] = sum0
                    h2emb1[i, j, k, m] = sum1

    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in range(8):
                    sum0 = sum1 = 0.0
                    for z in range(8):
                        sum0 += rot0_adj[i, z] * h2emb0[z, j, k, m]
                        sum1 += rot1_adj[i, z] * h2emb1[z, j, k, m]
                    h2emb0[i, j, k, m] = sum0
                    h2emb1[i, j, k, m] = sum1

    ##### get_iddt_cor1RDM check #####

    # indices: imp, bath
    # frag 0
    genfock0 = np.zeros((8, 8), dtype=complex)
    ifock0 = h1emb0
    corrden0, corrden20 = pyscf.fci.fci_dhf_slow.make_rdm12(ci0, 8, 4)

    for i in range(8):
        for j in range(8):
            for k in range(8):
                genfock0[j, i] += ifock0[i, k] * corrden0[k, j]
                for m in range(8):
                    for n in range(8):
                        temp = h2emb0[i, k, m, n] - h2emb0[m, k, i, n]
                        genfock0[j, i] += 0.5 * temp * corrden20[j, k, m, n]

    # frag 1
    genfock1 = np.zeros((8, 8), dtype=complex)
    ifock1 = h1emb1
    corrden1, corrden21 = pyscf.fci.fci_dhf_slow.make_rdm12(ci1, 8, 4)

    for i in range(8):
        for j in range(8):
            for k in range(8):
                genfock1[j, i] += ifock1[i, k] * corrden1[k, j]
                for m in range(8):
                    for n in range(8):
                        temp = h2emb1[i, k, m, n] - h2emb1[m, k, i, n]
                        genfock1[j, i] += 0.5 * temp * corrden21[j, k, m, n]

    # making TD 1RDM
    td_denfrag0 = np.zeros((8, 8), dtype=complex)
    td_denfrag1 = np.zeros((8, 8), dtype=complex)
    genfock0_adj = genfock0.transpose().conjugate()
    genfock1_adj = genfock1.transpose().conjugate()

    for i in range(8):
        for j in range(8):
            td_denfrag0[i, j] += genfock0[j, i] - genfock0_adj[j, i]
            td_denfrag1[i, j] += genfock1[j, i] - genfock1_adj[j, i]

    ##### get_glob1RDM check #####
    iglob = np.zeros((8, 8), dtype=complex)

    for p in range(8):
        for q in range(8):
            if p < 4 and q < 4:
                frag0 = np.dot(
                    rot0,
                    np.dot(td_denfrag0, rot0.conjugate().transpose()),
                )
                frag1 = np.dot(
                    rot0,
                    np.dot(td_denfrag0, rot0.conjugate().transpose()),
                )
                iglob[p, q] += 0.5j * (frag0[p, q] + frag1[p, q])
            if p < 4 and q >= 4:
                frag0 = np.dot(
                    rot0,
                    np.dot(td_denfrag0, rot0.conjugate().transpose()),
                )
                frag1 = np.dot(
                    rot1,
                    np.dot(td_denfrag1, rot1.conjugate().transpose()),
                )
                iglob[p, q] += 0.5j * (frag0[p, q] + frag1[p, q])
            if p >= 4 and q >= 4:
                frag0 = np.dot(
                    rot1,
                    np.dot(td_denfrag1, rot1.conjugate().transpose()),
                )
                frag1 = np.dot(
                    rot1,
                    np.dot(td_denfrag1, rot1.conjugate().transpose()),
                )
                iglob[p, q] += 0.5j * (frag0[p, q] + frag1[p, q])
            if p >= 4 and q < 4:
                frag0 = np.dot(
                    rot1,
                    np.dot(td_denfrag1, rot1.conjugate().transpose()),
                )
                frag1 = np.dot(
                    rot0,
                    np.dot(td_denfrag0, rot0.conjugate().transpose()),
                )
                iglob[p, q] += 0.5j * (frag0[p, q] + frag1[p, q])

    ##### get natural orbitals (in one_rk_step) #####

    noevals, noevecs = np.linalg.eigh(np.real(glob))
    # Re-order such that eigenvalues are in descending order
    noevals = np.flip(noevals)
    noevecs = np.flip(noevecs, 1)
    noevals = noevals.astype(complex)
    noevecs = noevecs.astype(complex)

    ##### get calc_Gmat #####

    gmat = np.zeros((8, 8), dtype=complex)

    for mu in range(8):
        for nu in range(8):
            for p in range(8):
                for q in range(8):
                    if mu != nu and np.abs(noevals[nu] - noevals[mu]) > 1e-9:
                        gmat[mu, nu] += (
                            np.conj(noevecs[p, mu]) * iglob[p, q] * noevecs[q, nu]
                        ) / (noevals[nu] - noevals[mu])
                    else:
                        gmat[mu, nu] = 0
    gmat_site = 1j * np.dot(noevecs, np.dot(gmat, np.conjugate(np.transpose(noevecs))))

    ##### checking get_ddt_mf_NOs #####

    td_noevecs = -1j * np.dot(gmat_site, noevecs)

    ##### checking get_ddt_mf1rdm_serial #####

    td_mf = -1j * (np.dot(gmat_site, mf) - np.dot(mf, gmat_site))

    ##### checking get_frag_Xmat #####

    xmat0 = np.zeros((8, 8), dtype=complex)
    xmat1 = np.zeros((8, 8), dtype=complex)

    mfevals0 = np.diag(np.real(np.dot(rot0.conjugate().transpose(), np.dot(mf, rot0))))
    mfevals1 = np.diag(np.real(np.dot(rot1.conjugate().transpose(), np.dot(mf, rot1))))

    for i in range(4, 8):
        for j in range(4, 8):
            for p in range(4, 8):
                for q in range(4, 8):
                    if i != j and np.abs(mfevals0[i] - mfevals0[j]) > 1e-9:
                        xmat0[j, i] += (
                            np.conj(rot0[p, j]) * td_mf[p, q] * rot0[q, i]
                        ) / (mfevals0[i] - mfevals0[j])
                    else:
                        xmat0[j, i] = 0

    for i in range(4, 8):
        for j in range(4, 8):
            for p in range(0, 4):
                for q in range(0, 4):
                    if i != j and np.abs(mfevals1[i] - mfevals1[j]) > 1e-9:
                        xmat1[j, i] += (
                            np.conj(rot1[p, j]) * td_mf[p, q] * rot1[q, i]
                        ) / (mfevals1[i] - mfevals1[j])
                    else:
                        xmat1[j, i] = 0

    xmat0 = -1j * xmat0
    xmat1 = -1j * xmat1

    ##### checking td of rotmat #####
    xmat0_site = np.dot(rot0, np.dot(xmat0, rot0.conjugate().transpose()))
    xmat1_site = np.dot(rot1, np.dot(xmat1, rot1.conjugate().transpose()))

    td_rot0 = -1j * np.dot(xmat0_site, rot0)
    td_rot1 = -1j * np.dot(xmat1_site, rot1)

    ##### checking new CI coefficients #####

    newh0 = h1emb0 - xmat0
    newh1 = h1emb1 - xmat1

    new_ci0 = -1j * applyham_pyscf.apply_ham_pyscf_spinor(
        ci0,
        newh0,
        h2emb0,
        4,
        8,
        0.0,
    )

    new_ci1 = -1j * applyham_pyscf.apply_ham_pyscf_spinor(
        ci1,
        newh1,
        h2emb1,
        4,
        8,
        0.0,
    )

    # factoring in timestep
    td_glob = iglob * delt
    td_mf = td_mf * delt
    td_noevecs = td_noevecs * delt
    td_rot0 = delt * td_rot0
    td_rot1 = delt * td_rot1
    td_ci0 = delt * new_ci0
    td_ci1 = delt * new_ci1

    # used for print_data
    corrdens = [corrden0, corrden1]
    corr2dens = [corrden20, corrden21]
    h1embs = [h1emb0, h1emb1]
    h2embs = [h2emb0, h2emb1]

    # check for MF 1RDM
    short_NOcc = np.copy(noevecs[:, : round(4)])
    short_ddtNOcc = np.copy(td_noevecs[:, : round(4)])
    chk = np.dot(short_ddtNOcc, short_NOcc.conj().T) + np.dot(
        short_NOcc, short_ddtNOcc.conj().T
    )

    ddtmf1RDM_check = np.allclose(chk, td_mf, rtol=0, atol=1e-5)

    return (
        td_noevecs,
        td_rot0,
        td_rot1,
        td_ci0,
        td_ci1,
        td_glob,
        td_mf,
        corrdens,
        corr2dens,
        h1embs,
        h2embs,
        ddtmf1RDM_check,
    )


for i in range(steps):
    # first step

    (
        del_NO1,
        del_rot01,
        del_rot11,
        del_ci01,
        del_ci11,
        del_glob1,
        del_mf1,
        corrdens,
        corr2dens,
        h1embs,
        h1embs,
        ddtmf1RDM_check,
    ) = one_rk_step(
        timestep,
        init_glob,
        init_mf,
        init_noevec,
        init_rot0,
        init_rot1,
        init_ci0,
        init_ci1,
        init_h1e,
        init_h2e,
    )

    if not ddtmf1RDM_check:
        print(f"td_mf failed at first step of step {i}")
        # exit()

    ### check how well natural orbitals diagonalize global rdm
    diag_global = utils.rot1el(init_glob, init_noevec)
    np.fill_diagonal(diag_global, 0)
    max_diag_global = np.max(np.abs(diag_global))
    # replacing the below with np.max(np.abs); not sure why return_max_value was created...
    # max_diag_global = self.return_max_value(diag_global)

    diag_globalRDM_check = np.allclose(
        diag_global,
        np.zeros((8, 8)),
        rtol=0,
        atol=1e-5,
    )

    if not diag_globalRDM_check:
        print(f"messed up at global check before first rk step of step {i}")
        exit()

    noevec = init_noevec + (0.5 * del_NO1)
    rot0 = init_rot0 + (0.5 * del_rot01)
    rot1 = init_rot1 + (0.5 * del_rot11)
    ci0 = init_ci0 + (0.5 * del_ci01)
    ci1 = init_ci1 + (0.5 * del_ci11)
    glob = init_glob + (0.5 * del_glob1)
    mf = init_mf + (0.5 * del_mf1)

    ### check how well natural orbitals diagonalize global rdm
    diag_global = utils.rot1el(glob, noevec)
    np.fill_diagonal(diag_global, 0)
    max_diag_global = np.max(np.abs(diag_global))
    # replacing the below with np.max(np.abs); not sure why return_max_value was created...
    # max_diag_global = self.return_max_value(diag_global)

    diag_globalRDM_check = np.allclose(
        diag_global,
        np.zeros((8, 8)),
        rtol=0,
        atol=1e-5,
    )

    if not diag_globalRDM_check:
        print(f"messed up at global check on first rk step of step {i}")
        print(f"max from check: {max_diag_global}")
        exit()

    # second step

    (
        del_NO2,
        del_rot02,
        del_rot12,
        del_ci02,
        del_ci12,
        del_glob2,
        del_mf2,
        corrdens,
        corr2dens,
        h1embs,
        h2embs,
        ddtmf1RDM_check,
    ) = one_rk_step(timestep, glob, mf, noevec, rot0, rot1, ci0, ci1, h1e, h2e)

    if not ddtmf1RDM_check:
        print(f"td_mf failed at second step of step {i}")
        # exit()

    noevec = init_noevec + (0.5 * del_NO2)
    rot0 = init_rot0 + (0.5 * del_rot02)
    rot1 = init_rot1 + (0.5 * del_rot12)
    ci0 = init_ci0 + (0.5 * del_ci02)
    ci1 = init_ci1 + (0.5 * del_ci12)
    glob = init_glob + (0.5 * del_glob2)
    mf = init_mf + (0.5 * del_mf2)

    # third step

    (
        del_NO3,
        del_rot03,
        del_rot13,
        del_ci03,
        del_ci13,
        del_glob3,
        del_mf3,
        corrdens,
        corr2dens,
        h1embs,
        h2embs,
        ddtmf1RDM_check,
    ) = one_rk_step(timestep, glob, mf, noevec, rot0, rot1, ci0, ci1, h1e, h2e)

    if not ddtmf1RDM_check:
        print(f"td_mf failed at third step of step {i}")
        # exit()

    noevec = init_noevec + del_NO3
    rot0 = init_rot0 + del_rot03
    rot1 = init_rot1 + del_rot13
    ci0 = init_ci0 + del_ci03
    ci1 = init_ci1 + del_ci13
    glob = init_glob + del_glob3
    mf = init_mf + del_mf3

    # fourth step

    (
        del_NO4,
        del_rot04,
        del_rot14,
        del_ci04,
        del_ci14,
        del_glob4,
        del_mf4,
        corrdens,
        corr2dens,
        h1embs,
        h2embs,
        ddtmf1RDM_check,
    ) = one_rk_step(timestep, glob, mf, noevec, rot0, rot1, ci0, ci1, h1e, h2e)

    if not ddtmf1RDM_check:
        print(f"td_mf failed at fourth step of step {i}")
        # exit()

    noevec = init_noevec + 1.0 / 6.0 * (
        del_NO1 + 2.0 * del_NO2 + 2.0 * del_NO3 + del_NO4
    )
    rot0 = init_rot0 + 1.0 / 6.0 * (
        del_rot01 + 2.0 * del_rot02 + 2.0 * del_rot03 + del_rot04
    )
    rot1 = init_rot1 + 1.0 / 6.0 * (
        del_rot11 + 2.0 * del_rot12 + 2.0 * del_rot13 + del_rot14
    )
    ci0 = init_ci0 + 1.0 / 6.0 * (del_ci01 + 2.0 * del_ci02 + 2.0 * del_ci03 + del_ci04)
    ci1 = init_ci1 + 1.0 / 6.0 * (del_ci11 + 2.0 * del_ci12 + 2.0 * del_ci13 + del_ci14)
    glob = init_glob + 1.0 / 6.0 * (
        del_glob1 + 2.0 * del_glob2 + 2.0 * del_glob3 + del_glob4
    )
    mf = init_mf + 1.0 / 6.0 * (del_mf1 + 2.0 * del_mf2 + 2.0 * del_mf3 + del_mf4)

    ### check how well natural orbitals diagonalize global rdm
    diag_global = utils.rot1el(glob, noevec)
    np.fill_diagonal(diag_global, 0)
    max_diag_global = np.max(np.abs(diag_global))
    # replacing the below with np.max(np.abs); not sure why return_max_value was created...
    # max_diag_global = self.return_max_value(diag_global)

    diag_globalRDM_check = np.allclose(
        diag_global,
        np.zeros((8, 8)),
        rtol=0,
        atol=1e-5,
    )

    if not diag_globalRDM_check:
        print(f"messed up at global check on step {i}")
        exit()

    init_noevec = noevec
    init_rot0 = rot0
    init_rot1 = rot1
    init_ci0 = ci0
    init_ci1 = ci1
    init_glob = glob
    init_mf = mf

    cicoeffs = [init_ci0, init_ci1]

    # Calculate total energy

    h_emb_halfcore0 = np.copy(h1embs[0][:8, :8])
    h_emb_halfcore1 = np.copy(h1embs[1][:8, :8])

    Efrag = 0
    for orb1 in range(4):
        for orb2 in range(8):
            Efrag += h_emb_halfcore0[orb1, orb2] * corrdens[0][orb2, orb1]
            Efrag += h_emb_halfcore1[orb1, orb2] * corrdens[1][orb2, orb1]
            for orb3 in range(8):
                for orb4 in range(8):
                    Efrag += 0.5 * (
                        h2embs[0][orb1, orb2, orb3, orb4]
                        * corr2dens[0][orb1, orb2, orb3, orb4]
                    )
                    Efrag += 0.5 * (
                        h2embs[1][orb1, orb2, orb3, orb4]
                        * corr2dens[1][orb1, orb2, orb3, orb4]
                    )

    # Etot = fci_mod.get_FCI_E(
    #        h1e,
    #        V_site,
    #        Ecore,
    #        cicoeffs,
    #        Nsites,
    #        int(Nelec / 2),
    #        int(Nelec / 2),
    #    )

    # Calculate total number of electrons
    # (used as convergence check for time-step)
    DMETNele = np.real(np.trace(corrdens[0][:2, :2]))
    DMETNele += np.real(np.trace(corrdens[1][2:, 2:]))

    # Print correlated density in the site basis
    corrdens_print = np.zeros(8)
    corrdens_print[:4] = np.copy(np.diag(np.real(corrdens[0][:4])))
    corrdens_print[4:] = np.copy(np.diag(np.real(corrdens[1][:4])))
    corrdens_print = corrdens_print.reshape(-1, 2).sum(axis=1)
    corrdens_print = np.insert(corrdens_print, 0, 0)  # last zero is current time
    file_corrdens = open("hardcode_corr_density.dat", "a")
    fmt_str = "%20.8e"
    np.savetxt(
        file_corrdens, corrdens_print.reshape(1, corrdens_print.shape[0]), fmt_str
    )
    file_corrdens.flush()

    # Print output data
    output = np.zeros(3)
    output[0] = i
    output[1] = Efrag
    output[2] = DMETNele
    file_output = open("hardcode_output_dynamics.dat", "a")
    np.savetxt(file_output, output.reshape(1, output.shape[0]), fmt_str)
    file_output.flush()
