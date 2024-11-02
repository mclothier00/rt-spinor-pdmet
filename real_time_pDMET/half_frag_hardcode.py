import numpy as np
import real_time_pDMET.scripts.utils as utils
import pyscf.fci
import real_time_pDMET.scripts.applyham_pyscf as applyham_pyscf

# four spatial sites, 8 spinors, 4 electrons, two fragments
# following contains ['glob1rdm', 'mf1rdm', 'NOevecs', 'rotmat_0', 'rotmat_1', 'CIcoeff_0', 'CIcoeff_1', 'h1e', h2e']
output = np.load("matrices.npz")
glob = output["glob1rdm"].astype(complex)
mf = output["mf1rdm"].astype(complex)
noevec = output["NOevecs"].astype(complex)
rot0 = output["rotmat_0"].astype(complex)
rot1 = output["rotmat_1"].astype(complex)
ci0 = output["CIcoeff_0"].astype(complex)
ci1 = output["CIcoeff_1"].astype(complex)

ham = np.load("ham.npz")
h1e = ham["h1e"]
h2e = ham["h2e"]
h1e = np.kron(np.eye(2, dtype=int), h1e)
h2e = utils.block_tensor(h2e)
h1e = utils.reshape_rtog_matrix(h1e)
h2e = utils.reshape_rtog_tensor(h2e)
h1e = h1e.astype(complex)
h2e = h2e.astype(complex)
long_fmt = False

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

f = open("output_halffrag_hardcode.txt", "w")
f.write("\n new rotmat, fragment 0: \n")
f.close()
utils.printarray(rot0, "output_halffrag_hardcode.txt", long_fmt)
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n new rotmat, fragment 1: \n")
f.close()
utils.printarray(rot1, "output_halffrag_hardcode.txt", long_fmt)


##### testing get_Hemb #####

# rotating 1e terms for fragment 1
h1emb0 = []
h1emb1 = []
rot0_adj = rot0.conjugate().transpose()
rot1_adj = rot1.conjugate().transpose()
h1emb0 = np.dot(rot0_adj, np.dot(h1e, rot0))
h1emb1 = np.dot(rot1_adj, np.dot(h1e, rot1))

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n h1 emb, frag 0, hard code: \n")
f.close()
utils.printarray(h1emb0, "output_halffrag_hardcode.txt", long_fmt)
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n h1 emb, frag 1, hard code: \n")
f.close()
utils.printarray(h1emb1, "output_halffrag_hardcode.txt", long_fmt)

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

f = open("output_halffrag_hardcode.txt", "a")
f.write(f"\n h2 emb, frag 0, hard code: \n {h2emb0[0,0,0,0]} \n shape: {h2emb0.shape}")
f.write(
    f"\n h2 emb, frag 1, hard code: \n {h2emb1[0,0,0,0]} \n shape: {h2emb1.shape} \n"
)
f.close()

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

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n fock, frag 0 (above) \n")
f.close
utils.printarray(genfock0, "output_halffrag_hardcode.txt", long_fmt)
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n fock, frag 1 (above) \n")
f.close
utils.printarray(genfock1, "output_halffrag_hardcode.txt", long_fmt)

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n TD of corr1RDM, frag 0 (above) \n")
f.close
utils.printarray(td_denfrag0, "output_halffrag_hardcode.txt", long_fmt)
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n TD of corr1RDM, frag 1 (above) \n")
f.close
utils.printarray(td_denfrag1, "output_halffrag_hardcode.txt", long_fmt)


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

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n TD of global1RDM: \n")
f.close()
utils.printarray(iglob, "output_halffrag_hardcode.txt", long_fmt)


##### get natural orbitals (in one_rk_step) #####
noevals, noevecs = np.linalg.eigh(np.real(glob))

# Re-order such that eigenvalues are in descending order
noevals = np.flip(noevals)
noevecs = np.flip(noevecs, 1)
noevals = noevals.astype(complex)
noevecs = noevecs.astype(complex)

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n NO eigenvectors (U): \n")
f.close()
utils.printarray(noevecs, "output_halffrag_hardcode.txt", long_fmt)

##### get calc_Gmat #####

gmat = np.zeros((8, 8), dtype=complex)

for mu in range(8):
    for nu in range(8):
        for p in range(8):
            for q in range(8):
                if mu != nu and np.abs(noevals[nu] - noevals[mu]) > 1e-5:
                    gmat[mu, nu] += (
                        np.conj(noevecs[p, mu]) * iglob[p, q] * noevecs[q, nu]
                    ) / (noevals[nu] - noevals[mu])
                else:
                    gmat[mu, nu] = 0

# switch to site basis for following calculations
gmat_site = 1j * np.dot(noevecs, np.dot(gmat, np.conjugate(np.transpose(noevecs))))

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n G matrix: \n")
f.close()
utils.printarray(gmat_site, "output_halffrag_hardcode.txt", long_fmt)


##### checking get_ddt_mf_NOs #####

td_noevecs = -1j * np.dot(gmat_site, noevecs)

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n TD natural orbitals (U dot): \n")
f.close()
utils.printarray(td_noevecs, "output_halffrag_hardcode.txt", long_fmt)


##### checking get_ddt_mf1rdm_serial #####

td_mf = -1j * (np.dot(gmat_site, mf) - np.dot(mf, gmat_site))

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n TD mean field 1RDM: \n")
f.close()
utils.printarray(td_mf, "output_halffrag_hardcode.txt", long_fmt)

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
                    xmat0[j, i] += (np.conj(rot0[p, j]) * td_mf[p, q] * rot0[q, i]) / (
                        mfevals0[i] - mfevals0[j]
                    )
                else:
                    xmat0[j, i] = 0

for i in range(4, 8):
    for j in range(4, 8):
        for p in range(0, 4):
            for q in range(0, 4):
                if i != j and np.abs(mfevals1[i] - mfevals1[j]) > 1e-9:
                    xmat1[j, i] += (np.conj(rot1[p, j]) * td_mf[p, q] * rot1[q, i]) / (
                        mfevals1[i] - mfevals1[j]
                    )
                else:
                    xmat1[j, i] = 0

xmat0 = -1j * xmat0
xmat1 = -1j * xmat1

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n X matrix, fragment 0: \n")
f.close()
utils.printarray(xmat0, "output_halffrag_hardcode.txt", long_fmt)
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n X matrix, fragment 1: \n")
f.close()
utils.printarray(xmat1, "output_halffrag_hardcode.txt", long_fmt)


##### checking td of rotmat #####

xmat0_site = np.dot(rot0, np.dot(xmat0, np.conjugate(np.transpose(rot0))))
xmat1_site = np.dot(rot1, np.dot(xmat1, np.conjugate(np.transpose(rot1))))

td_rot0 = -1j * np.dot(xmat0_site, rot0)
td_rot1 = -1j * np.dot(xmat1_site, rot1)

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n new rotmat, fragment 0: \n")
f.close()
utils.printarray(td_rot0, "output_halffrag_hardcode.txt", long_fmt)
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n new rotmat, fragment 1: \n")
f.close()
utils.printarray(td_rot1, "output_halffrag_hardcode.txt", long_fmt)


##### checking new CI coefficients #####

newh0 = h1emb0 - xmat0
newh1 = h1emb1 - xmat1

delt = 0.001

new_ci0 = (
    -1j
    * delt
    * applyham_pyscf.apply_ham_pyscf_spinor(
        ci0,
        newh0,
        h2emb0,
        4,
        8,
        0.0,
    )
)
new_ci1 = (
    -1j
    * delt
    * applyham_pyscf.apply_ham_pyscf_spinor(
        ci1,
        newh1,
        h2emb1,
        4,
        8,
        0.0,
    )
)

f = open("output_halffrag_hardcode.txt", "a")
f.write("\n new CI coefficients, fragment 0: \n")
f.close()
utils.printarray(new_ci0, "output_halffrag_hardcode.txt")
f = open("output_halffrag_hardcode.txt", "a")
f.write("\n new ci coefficients, fragment 1: \n")
f.close()
utils.printarray(new_ci1, "output_halffrag_hardcode.txt")


# tracking observables
# NOTE: check that td_denfrag is the correct matrix
corrdens = np.zeros(8)
corrdens[:4] = np.copy(np.diag(np.real(corrden0[:4])))
corrdens[4:] = np.copy(np.diag(np.real(corrden1[:4])))
corrdens = corrdens.reshape(-1, 2).sum(axis=1)
corrdens = np.insert(corrdens, 0, 0)  # last zero is current time
print(f"density matrix: {corrdens}")

# supposed to be frag.corr1RDM?
DMETNele = np.real(np.trace(corrden0[:2, :2]))
DMETNele += np.real(np.trace(corrden1[2:, 2:]))
print(f"number of electrons: {DMETNele}")


# need to do this for both fragments...

# same as h1emb, because no core orbitals
h_emb_halfcore0 = np.copy(h1emb0[:8, :8])
h_emb_halfcore1 = np.copy(h1emb1[:8, :8])

Efrag = 0
for orb1 in range(4):
    for orb2 in range(8):
        Efrag += h_emb_halfcore0[orb1, orb2] * corrden0[orb2, orb1]
        Efrag += h_emb_halfcore1[orb1, orb2] * corrden1[orb2, orb1]
        for orb3 in range(8):
            for orb4 in range(8):
                Efrag += 0.5 * (
                    h2emb0[orb1, orb2, orb3, orb4] * corrden20[orb1, orb2, orb3, orb4]
                )
                Efrag += 0.5 * (
                    h2emb1[orb1, orb2, orb3, orb4] * corrden21[orb1, orb2, orb3, orb4]
                )

print(f"dimension of h2emb: {h2emb1.shape}")
print(f"dimension of corrden21: {corrden21.shape}")
print(f"total energy: {Efrag}")

np.savez(
    "4site_hardcode.npz",
    rot0=rot0,
    rot1=rot1,
    h1emb0=h1emb0,
    h1emb1=h1emb1,
    genfock0=genfock0,
    genfock1=genfock1,
    td_denfrag0=td_denfrag0,
    td_denfrag1=td_denfrag1,
    td_glob=iglob,
    noevecs=noevecs,
    gmat=gmat,
    td_noevecs=td_noevecs,
    td_mf=td_mf,
    xmat0=xmat0,
    xmat1=xmat1,
    td_rot0=td_rot0,
    td_rot1=td_rot1,
)
