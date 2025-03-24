import numpy as np
import time
import scipy.linalg as la
import real_time_pDMET.rtpdmet.static.fragment_mod as fragment_mod
from pyscf import gto, scf, ao2mo
from real_time_pDMET.rtpdmet.static.quad_fit import quad_fit_mu
from math import copysign
from pyscf import lib
from mpi4py import MPI

DiisDim = 4
adiis = lib.diis.DIIS()
adiis.space = DiisDim


class static_pdmet:
    def __init__(
        self,
        Nsites,
        Nele,
        Nfrag,
        impindx,
        h_site,
        V_site,
        U,
        Maxitr,
        mf1RDM,
        tol,
        hamtype=0,
        mubool=False,
        muhistory=False,
        hubb_indx=None,
        nelecTol=1e-5,
        dmu=0.02,
        step=0.05,
        trust_region=2.5,
    ):
        """
        Nsites    - total number of sites (or basis functions) in total system
        Nele      - total number of electrons
        Nfrag     - total number of fragments for DMET calculation
        impindx   - a list of numpy arrays containing the impurity indices for
                    each fragment
        h_site    - 1 e- hamiltonian in site-basis for total system
        V-site    - 2 e- hamiltonian in site-basis for total system
        mubool    - boolean switching between using a global chem potential to
                    optimize DMET # of electrons or not
        muhistory - boolean switch for chemical potential fitting
        (if true = use historic information, if false = use quadratic fit and
                    linear regression)
        maxitr    - max number of DMET iterations#maxitr  - max number of DMET
                    iterations
        tol       - tolerance for difference in 1 RDM during DMET cycle
        U         - Hubbard constant for electron interactions
        """

        print()
        print("***************************************")
        print("         INITIALIZING PDMET          ")
        print("***************************************")
        print()

        self.mubool = mubool
        self.muhistory = muhistory
        self.trust_region = trust_region
        self.dmu = dmu
        self.step = step
        self.Maxitr = Maxitr
        self.tol = tol
        self.nelecTol = nelecTol
        self.h_site = h_site
        self.V_site = V_site
        self.hamtype = hamtype
        self.hubb_indx = hubb_indx
        self.U = U
        self.Nsites = Nsites
        self.Nele = Nele
        self.Nfrag = Nfrag
        self.mu = 0
        self.DiisStart = 4
        self.DiisDim = 4
        self.history = []

        # Calculate an initial mean-field Hamiltonian
        
        print("Calculating initial mean-field Hamiltonian")
        if hamtype == 0:
            if mf1RDM is None:
                mf1RDM = self.initialize_RHF(h_site, V_site)
                self.old_glob1RDM = np.copy(mf1RDM)
                # mf1RDM = hf.hubbard_1RDM(self.Nele, h_site)
                # mf1RDM = self.initialize_UHF(h_site, V_site)

            else:
                self.old_glob1RDM = np.copy(mf1RDM)

        else:
            if mf1RDM is None:
                mf1RDM = self.initialize_RHF(h_site, V_site)
                self.old_glob1RDM = np.copy(mf1RDM)
                # mf1RDM = hf.hubbard_1RDM(self.Nele, h_site)
                # mf1RDM = self.initialize_UHF(h_site, V_site)
            else:
                self.old_glob1RDM = np.copy(mf1RDM)

        self.mf1RDM = mf1RDM

        # Initialize the system from mf 1RDM and fragment information

        self.frag_list = []
        for i in range(Nfrag):
            self.frag_list.append(
                fragment_mod.fragment(impindx[i], Nsites, Nele, hubb_indx)
            )
            self.frag_list[i].frag_num = i

        # list that takes site index and gives
        # fragment index corresponding to that site

        self.site_to_frag_list = []
        self.site_to_impindx = []
        for i in range(Nsites):
            for ifrag, array in enumerate(impindx):
                if i in array:
                    self.site_to_frag_list.append(ifrag)
        self.site_to_impindx.append(np.argwhere(array == i)[0][0])

        # output file
        self.file_output = open("output_static.dat", "w")

        # Parallelization

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        size = comm.Get_size()

        frag_per_rank = []
        for i in range(size):
            frag_per_rank.append([])

        # this currently doesn't do anything 
        self.site_to_frag = []

        if self.rank == 0:
            for i, frag in enumerate(self.frag_list):
                frag_per_rank[i % size].append(frag)
                self.frag_list = None
            self.frag_in_rank = frag_per_rank[0]
            for r, frag in enumerate(frag_per_rank):
                if r != 0:
                    comm.send(frag, dest=r)

            for frag in self.frag_in_rank:
                for j in frag.impindx:
                    self.site_to_frag.append(self.site_to_frag_list[j])

        else:
            self.frag_in_rank = comm.recv(source=0)
            self.frag_list = None
            for frag in self.frag_in_rank:
                for j in frag.impindx:
                    self.site_to_frag.append(self.site_to_frag_list[j])

        for i, frag in enumerate(self.frag_in_rank):
            frag.frags_rank = comm.Get_rank()

    ##########################################################

    def kernel(self):
        # Initialize the system from mf 1RDM and fragment information

        if self.rank == 0:
            print()
            print("***************************************")
            print("    BEGIN STATIC PDMET CALCULATION     ")
            print("***************************************")

        # DMET outer loop
        start_time = time.perf_counter()
        dVcor_per_ele = None
        conv = False
        old_E = 0.0
        old_glob1RDM = np.copy(self.old_glob1RDM)

        for itr in range(self.Maxitr):
            
            if self.rank == 0:
                print()
                print("Iteration:", itr)
                print()
            
            # embedding calculation
            if self.mubool:
                # do correlation calculation and add the self.mu to the H_emb
                totalNele_0 = self.corr_calc_with_mu(self.mu)
                record = [(0.0, totalNele_0)]

                if abs((totalNele_0 / self.Nele) - 1.0) < self.nelecTol:
                    print(f"chemical potential fitting is unnecessary on rank {self.rank}")
                    total_Nele = totalNele_0
                    self.history.append(record)

                else:
                    if self.muhistory:
                        # predict from  historic information
                        temp_dmu = self.predict(totalNele_0, self.Nele)
                        print(f"temp_dmu from prediction on rank {self.rank}: {temp_dmu}")
                        if temp_dmu is not None:
                            self.dmu = temp_dmu
                        else:
                            self.dmu = abs(self.dmu) * (
                                -1 if (totalNele_0 > self.Nele) else 1
                            )
                    else:
                        self.dmu = abs(self.dmu) * (
                            -1 if (totalNele_0 > self.Nele) else 1
                        )
                    print(f"chemical potential dmu after 1st approximation on rank {self.rank}: {self.dmu}")

                    test_mu = self.mu + self.dmu
                    totalNele_1 = self.corr_calc_with_mu(test_mu)
                    record.append((self.dmu, totalNele_1))

                    if abs((totalNele_1 / self.Nele) - 1.0) < self.nelecTol:
                        print(f"chemical potential is converged on rank {self.rank} with dmu: {self.dmu}")
                        self.history.append(record)
                        self.mu = test_mu
                        total_Nele = totalNele_1

                    else:
                        Neleprime = (totalNele_1 - totalNele_0) / self.dmu
                        dmu1 = (self.Nele - totalNele_0) / Neleprime
                        if abs(dmu1) > self.step:
                            print(
                                "extrapolation dmu",
                                dmu1,
                                "is greater then the step ",
                                self.step,
                            )
                            dmu1_tmp = copysign(self.step, dmu1)
                            self.step = min(abs(dmu1), 0.25)
                            dmu1 = dmu1_tmp

                        test_mu = self.mu + dmu1
                        totalNele_2 = self.corr_calc_with_mu(test_mu)
                        record.append((dmu1, totalNele_2))

                        if abs((totalNele_2 / self.Nele) - 1.0) < self.nelecTol:
                            print(f"chem potential is converged on rank {self.rank} w/ dmu1: {dmu1}")
                            self.mu = test_mu
                            self.history.append(record)
                            total_Nele = totalNele_2

                        else:
                            mus = np.array([0.0, self.dmu, dmu1])
                            Neles = np.array([totalNele_0, totalNele_1, totalNele_2])
                            dmu2 = quad_fit_mu(mus, Neles, self.Nele / 2, self.step)

                            test_mu = self.mu + dmu2
                            totalNele_3 = self.corr_calc_with_mu(test_mu)
                            record.append((dmu2, totalNele_3))

                            if abs(totalNele_3 / self.Nele - 1.0) < self.nelecTol:
                                print(f"chem potential is converged on rank {self.rank} w/ dmu2: {dmu2}")
                                self.mu = test_mu
                                self.history.append(record)
                                total_Nele = totalNele_3
                            else:
                                mus = np.array([0.0, self.dmu, dmu1, dmu2])
                                Neles = np.array(
                                    [totalNele_0, totalNele_1, totalNele_2, totalNele_3]
                                )
                                dmu3 = quad_fit_mu(mus, Neles, self.Nele / 2, self.step)

                                test_mu = self.mu + dmu3
                                totalNele_4 = self.corr_calc_with_mu(test_mu)
                                print(
                                    f"mu didnt converge on rank {self.rank}, final electron #: {totalNele_4}"
                                )
                                record.append((dmu3, totalNele_4))
                                total_Nele = totalNele_4
                                self.history.append(record)
                                self.mu = test_mu

            else:
                if self.rank == 0:
                    print("No chemical potential fitting is employed")

                for frag in self.frag_in_rank:
                    frag.corr_calc(
                        self.mf1RDM,
                        self.h_site,
                        self.V_site,
                        self.U,
                        self.mu,
                        self.hamtype,
                        self.hubb_indx,
                        self.mubool,
                    )

            # constract a global density matrix from all impurities
            self.get_globalRDM()

            # DIIS routine
            if itr >= self.DiisStart:
                self.glob1RDM = adiis.update(self.glob1RDM)
            dif = np.linalg.norm(self.glob1RDM - old_glob1RDM)
            dVcor_per_ele = self.max_abs(dif)
            old_glob1RDM = np.copy(self.glob1RDM)

            # getting natural orbitals from eigenvectors of the global rdm
            self.get_nat_orbs()
            self.get_new_mfRDM(int(self.Nele / 2))
            self.get_DMET_E()
            total_Nele = self.just_Nele()
            if old_E is None:
                old_E = np.copy(self.DMET_E)

            dE = self.DMET_E - old_E
            old_E = np.copy(self.DMET_E)

            if np.mod(itr, self.Maxitr / 100) == 0 and itr > 0:
                if self.rank == 0:
                    print("Finished DMET Iteration", itr)
                    print("Current difference in global 1RDM =", dif)
                    print("vcore=", dVcor_per_ele)
                    self.calc_data(itr, dif, total_Nele)

            if dVcor_per_ele < self.tol and abs(dE) < 1.0e-6:
                conv = True
                break

        if self.rank == 0:
            print()
            print("***************************************")
            print("    FINISH STATIC PDMET CALCULATION    ")
            print("***************************************")
            print()
            print("Final DMET energy =", self.DMET_E)
            print("Energy per site for U=", self.U, "is:", (self.DMET_E / self.Nsites))
            if conv:
                print("DMET calculation succesfully converged in", itr, "iterations")
                print("Final difference in global 1RDM =", dif)
                print()

            else:
                print(
                    "WARNING:DMET calculation finished, but did not converge in",
                    self.Maxitr,
                    "iterations",
                )
                print("Final difference in global 1RDM =", dif)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        if self.rank == 0:
            print("total_time", total_time)
            self.file_output.close()

    ##########################################################

    def max_abs(self, x):
        # Equivalent to np.max(np.abs(x)), but faster.
        if np.iscomplexobj(x):
            return np.abs(x).max()
        else:
            return max(np.max(x), abs(np.min(x)))

    ##########################################################

    def initialize_UHF(self, h_site, V_site):
        Norbs = self.Nele
        mol = gto.M()
        mol.nelectron = self.Nele
        mol.imncore_anyway = True
        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h_site
        mf.get_ovlp = lambda *args: np.eye(Norbs)
        mf._eri = ao2mo.restore(8, V_site, Norbs)
        # evals, h = np.linalg.eigh(h_site)
        # mf.init_guess = h

        mf.kernel()
        mfRDM = mf.make_rdm1()

        return mfRDM

    ##########################################################

    def initialize_RHF(self, h_site, V_site):
        print("Mf 1RDM is initialized with RHF")
        Norbs = self.Nele
        mol = gto.M()
        mol.nelectron = self.Nele
        mol.imncore_anyway = True
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h_site
        mf.get_ovlp = lambda *args: np.eye(Norbs)
        mf._eri = ao2mo.restore(8, V_site, Norbs)

        mf.kernel()
        mfRDM = mf.make_rdm1()

        return mfRDM

    ##########################################################

    def initialize_GHF(self, h_site, V_site):
        print("Mf 1RDM is initialized with GHF")
        Norbs = self.Nele
        mol = gto.M()
        mol.nelectron = self.Nele
        mol.imncore_anyway = True
        mf = scf.GHF(mol)
        mf.get_hcore = lambda *args: h_site
        mf.get_ovlp = lambda *args: np.eye(Norbs)
        mf._eri = ao2mo.restore(8, V_site, Norbs)

        mf.kernel()
        mfRDM = mf.make_rdm1()

        return mfRDM

    ##########################################################

    def get_globalRDM(self):
        # initialize glodal 1RDM to be complex if rotation
        # matrix or correlated 1RDM is complex

        self.glob1RDM = np.zeros([self.Nsites, self.Nsites])

        # form the global 1RDM forcing hermiticity
        self.globalRDMtrace = 0

        mpi_glob1RDM = np.zeros([self.Nsites, self.Nsites])

        for i, frag in enumerate(self.frag_in_rank):
            # ordered as impurity, virtual, bath, core to match rotmat
            fullcorr1RDM = np.zeros((self.Nsites, self.Nsites))
            #fullcorr1RDM[frag.imprange, frag.imprange] = frag.corr1RDM[:frag.Nimp, :frag.Nimp]
            # impurity
            fullcorr1RDM[:frag.Nimp, :frag.Nimp] = frag.corr1RDM[:frag.Nimp, :frag.Nimp]
            # bath
            fullcorr1RDM[frag.last_virt:frag.last_bath, frag.last_virt:frag.last_bath] = frag.corr1RDM[frag.Nimp:, frag.Nimp:]
            # impurity-bath coupling
            fullcorr1RDM[frag.last_virt:frag.last_bath, :frag.Nimp] = frag.corr1RDM[frag.Nimp:, :frag.Nimp]
            fullcorr1RDM[:frag.Nimp, frag.last_virt:frag.last_bath] = frag.corr1RDM[:frag.Nimp, frag.Nimp:]
            # core
            fullcorr1RDM[frag.last_bath:, frag.last_bath:] = 2 * np.eye(frag.Ncore)
            tmp = 0.5 * np.dot(
                frag.rotmat, np.dot(fullcorr1RDM, frag.rotmat.conj().T)
            )
            #print(f'tmp for rank {self.rank}: \n {tmp}')
            for site in frag.impindx:
                mpi_glob1RDM[site, :] += tmp[site, :]
                mpi_glob1RDM[:, site] += tmp[:, site]

        #print(f'mpi global matrix on rank {self.rank}: \n {mpi_glob1RDM}')

        self.glob1RDM = np.zeros([self.Nsites, self.Nsites])
        MPI.COMM_WORLD.Allreduce(mpi_glob1RDM, self.glob1RDM, op=MPI.SUM)
        trace1RDM = self.glob1RDM.trace()

    ##########################################################

    def predict(self, nelec, target):
        """
        assume the chemical potential landscape more
        or less the same for revious fittings.
        the simplest thing to do is predicting
        a dmu from each previous fitting, and compute
        a weighted average. The weight should prefer
        lattest runs, and prefer the fittigs that
        has points close to current and target number of electrons

        """
        from math import sqrt, exp

        vals = []
        weights = []

        # hyperparameters
        damp_factor = np.e
        sigma2, sigma3 = 0.00025, 0.0005

        for dmu, record in enumerate(self.history):
            # exponential
            weight = damp_factor ** (dmu + 1 - len(self.history))

            if len(record) == 1:
                val, weight = 0.0, 0.0
                continue

            elif len(record) == 2:
                # fit a line
                (mu1, n1), (mu2, n2) = record
                slope = (n2 - n1) / (mu2 - mu1)
                val = (target - nelec) / slope
                # weight factor
                metric = min(
                    (target - n1) ** 2 + (nelec - n2) ** 2,
                    (target - n2) ** 2 + (nelec - n1) ** 2,
                )
                # Gaussian weight
                weight *= exp(-0.5 * metric / sigma2)

            elif len(record) == 3:
                # check to make sure that the data is monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(record)
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0.0, 0.0
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous

                a, b, c = np.dot(
                    la.inv(
                        np.asarray(
                            [[mu1**2, mu1, 1], [mu2**2, mu2, 1], [mu3**2, mu3, 1]]
                        )
                    ),
                    np.asarray([n1, n2, n3]).reshape(-1, 1),
                ).reshape(-1)

                # if the parabola is not monotonic,
                # use linear interpolation instead
                if mu1 < -0.5 * b / a < mu3:

                    def find_mu(n):
                        if n < n2:
                            slope = (n2 - n1) / (mu2 - mu1)
                        else:
                            slope = (n2 - n3) / (mu2 - mu3)
                        return mu2 + (n - n2) / slope

                else:

                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n - n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n - n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c - n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                    (target - n1) ** 2 + (nelec - n2) ** 2,
                    (target - n1) ** 2 + (nelec - n3) ** 2,
                    (target - n2) ** 2 + (nelec - n1) ** 2,
                    (target - n2) ** 2 + (nelec - n3) ** 2,
                    (target - n3) ** 2 + (nelec - n1) ** 2,
                    (target - n3) ** 2 + (nelec - n2) ** 2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            else:  # len(record) >= 4:
                # first find three most nearest points
                mus, nelecs = zip(*record)
                mus = np.asarray(mus)
                nelecs = np.asarray(nelecs)
                delta_nelecs = np.abs(nelecs - target)
                idx_dN = np.argsort(delta_nelecs, kind="mergesort")
                mus_sub = mus[idx_dN][:3]
                nelecs_sub = nelecs[idx_dN][:3]

                # we need to check data sanity: should be monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(zip(mus_sub, nelecs_sub))
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0.0, 0.0
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous
                a, b, c = np.dot(
                    la.inv(
                        np.asarray(
                            [[mu1**2, mu1, 1], [mu2**2, mu2, 1], [mu3**2, mu3, 1]]
                        )
                    ),
                    np.asarray([n1, n2, n3]).reshape(-1, 1),
                ).reshape(-1)

                # if the parabola is not monotonic,
                # use linear interpolation instead
                if mu1 < -0.5 * b / a < mu3:

                    def find_mu(n):
                        if n < n2:
                            slope = (n2 - n1) / (mu2 - mu1)
                        else:
                            slope = (n2 - n3) / (mu2 - mu3)
                        return mu2 + (n - n2) / slope

                else:

                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n - n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n - n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c - n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                    (target - n1) ** 2 + (nelec - n2) ** 2,
                    (target - n1) ** 2 + (nelec - n3) ** 2,
                    (target - n2) ** 2 + (nelec - n1) ** 2,
                    (target - n2) ** 2 + (nelec - n3) ** 2,
                    (target - n3) ** 2 + (nelec - n1) ** 2,
                    (target - n3) ** 2 + (nelec - n2) ** 2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            vals.append(val)
            weights.append(weight)

        if np.sum(weights) > 1e-3:
            dmu = np.dot(vals, weights) / np.sum(weights)
            if abs(dmu) > 0.5:
                dmu = copysign(0.5, dmu)
            print("adaptive chemical potential fitting, dmu =", dmu)
            return dmu
        else:
            print("adaptive chemical potential fitting not used")
            return None

    ##########################################################

    def corr_calc_with_mu(self, mu):
        rankNele = 0.0
        for frag in self.frag_in_rank:
            fragNele = frag.corr_calc(
                self.mf1RDM,
                self.h_site,
                self.V_site,
                self.U,
                mu,
                self.hamtype,
                self.hubb_indx,
                self.mubool,
            )
            rankNele += fragNele
        
        totalNele = MPI.COMM_WORLD.allreduce(rankNele, op=MPI.SUM)
        print("total electrons:", totalNele)
        
        return totalNele

    ##########################################################

    def get_Nele(self, mu):
        totalNele = 0.0
        for frag in self.frag_in_rank:
            frag.add_mu_Hemb(mu)
            frag.solve_GS(self.U)
            frag.get_corr1RDM()
            fragNele = frag.nele_in_frag()
            totalNele += fragNele
            new_mu = -1 * mu
            # to make sure Im not changing
            # H_emb with wrong guess for dmu
            frag.add_mu_Hemb(new_mu)
        print("total electrons:", totalNele)
        return totalNele

    ##########################################################

    def just_Nele(self):
        totalNele = 0.0
        for frag in self.frag_in_rank:
            fragNele = frag.nele_in_frag()
            totalNele += fragNele
        return totalNele

    ##########################################################

    def get_nat_orbs(self):
        NOevals, NOevecs = np.linalg.eigh(self.glob1RDM)
        # Re-order such that eigenvalues are in descending order
        self.NOevals = np.flip(NOevals)
        self.NOevecs = np.flip(NOevecs, 1)

    ##########################################################

    def get_new_mfRDM(self, NOcc):
        # get mf 1RDM from the first Nocc
        # natural orbitals of the global rdm
        # (natural orbitals with the highest occupation)
        NOcc = self.NOevecs[:, :NOcc]
        self.mf1RDM = 2.0 * np.dot(NOcc, NOcc.T.conj())

    ##########################################################

    def get_frag_corr12RDM(self):
        # correlated 1 RDM for each fragment
        for frag in self.frag_in_rank:
            frag.get_corr12RDM()

    ##########################################################

    def get_frag_Hemb(self):
        # Hamiltonian for each fragment
        for frag in self.frag_in_rank:
            frag.get_Hemb(
                self.h_site, self.V_site, self.U, self.hamtype, self.hubb_indx
            )

    ##########################################################

    def Hemb_add_mu(self, mu):
        for frag in self.frag_in_rank:
            frag.add_mu_Hemb(mu)

    ##########################################################

    def calc_data(self, itr, dif, total_Nele):
        fmt_str = "%20.8e"
        output = np.zeros(6 + self.Nsites)
        output[0] = itr
        output[1] = self.mu
        output[2] = dif
        output[3] = total_Nele
        output[4] = self.DMET_E / self.Nsites
        output[5 : 5 + self.Nsites] = self.NOevals
        np.savetxt(self.file_output, output.reshape(1, output.shape[0]), fmt_str)
        self.file_output.flush()
        np.save("mfRDM_static", self.mf1RDM)
        np.save("globRDM_static", self.glob1RDM)
        CI = []
        rotmat = []
        for frag in self.frag_in_rank:
            CI.append(np.copy(frag.CIcoeffs))
            rotmat.append(np.copy(frag.rotmat))
        np.save("CI_ststic", CI)

    ##########################################################

    def get_DMET_E(self):
        self.get_frag_Hemb()
        self.get_frag_corr12RDM()
        self.DMET_E = 0.0
        for frag in self.frag_in_rank:
            frag.get_frag_E()
            self.DMET_E += np.real(frag.Efrag)
            # discard what should be numerical error of imaginary part
