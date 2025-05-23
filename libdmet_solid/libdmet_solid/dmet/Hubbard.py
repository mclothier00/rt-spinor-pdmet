#! /usr/bin/env python

import os
import numpy as np
from math import copysign

from pyscf import lib

from libdmet_solid.dmet.HubPhSymm import *
from libdmet_solid.routine.slater_helper import transform_imp
from libdmet_solid.dmet.quad_fit import quad_fit_mu
from libdmet_solid import utils

def HartreeFock(Lat, v, filling, mu0=None, beta=np.inf, ires=False, **kwargs):
    """
    RHF and UHF wrapper.
    """
    restricted = v.restricted
    if beta != np.inf:
        log.info("using finite-T smearing for lattice, beta = %20.12f ", beta)
    rho, mu, E, res = HF(Lat, v, filling, restricted, mu0=mu0, beta=beta, ires=True, **kwargs)
    log.result("Local density matrix (mean-field): \n%s", \
            rho[:, 0])
    log.result("Chemical potential (mean-field) = %s", mu)
    log.result("Energy per cell (mean-field) = %20.12f", E)
    log.result("Gap (mean-field) = %s" % res["gap"])
    if ires:
        return rho, mu, res
    else:
        return rho, mu

def RHartreeFock(Lat, v, filling, mu0=None, beta=np.inf, ires=False, **kwargs):
    log.eassert(v.restricted == True, "RHF routine requires vcor is restricted.")
    return HartreeFock(Lat, v, filling, mu0=mu0, beta=beta, ires=ires, **kwargs)

def transformResults(rhoEmb, E, basis, ImpHam, H1e=None, int_bath=False, **kwargs):
    print("!!!!!!!!1 Transform results in hubbard!!!!!!!!!")
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    rhoImp, Efrag, nelec = slater.transformResults(rhoEmb, E, basis, ImpHam, H1e, **kwargs)
    log.debug(1, "impurity density matrix:\n%s", rhoImp)
    print("Efrag in Hubbard", Efrag)
    if Efrag is None:

        return nelec / nscsites

    else:
        if int_bath:
            # explicitly take out lattice and last_dmu, avoid duplicates in args
            lattice = kwargs.pop("lattice")
            last_dmu = kwargs.pop("last_dmu")
            Efrag = slater.get_E_dmet(basis, lattice, ImpHam, last_dmu, \
                        rdm1_emb=rhoEmb, **kwargs)
        log.result("Local density matrix (impurity):")
        for s in range(spin):
            log.result("%s", rhoImp[s])
        log.result("nelec per cell (impurity) = %20.12f", nelec)
        log.result("Energy per cell (impurity) = %20.12f", Efrag)
        return rhoImp, Efrag / nscsites, nelec / nscsites

def transformResults_new(rhoEmb, E, lattice, basis, ImpHam, H1e, last_dmu, int_bath=False, **kwargs):
    log.warn("transformResults_new is being deprecated.\nUse transformResults instead.")
    kwargs["lattice"] = lattice
    kwargs["last_dmu"] = last_dmu
    return transformResults(rhoEmb, E, basis, ImpHam, H1e=H1e, int_bath=int_bath, **kwargs)

def apply_dmu(lattice, ImpHam, basis, dmu, **kwargs):
    #nscsites = lattice.nscsites
    #nbasis = basis.shape[-1]
    #if "dmu_idx" in kwargs and kwargs["dmu_idx"] is not None:
    #    dmu_idx = kwargs["dmu_idx"]
    #else:
    #    dmu_idx = range(nscsites)
    #dmu_mat = np.zeros((nscsites, nscsites))
    #dmu_mat[dmu_idx, dmu_idx] = dmu

    #if ImpHam.restricted:
    #    ImpHam.H1["cd"][0] -= transform_imp(basis[0], lattice, dmu_mat)
    #else:
    #    ImpHam.H1["cd"][0] -= transform_imp(basis[0], lattice, dmu_mat)
    #    ImpHam.H1["cd"][1] -= transform_imp(basis[1], lattice, dmu_mat)
    #ImpHam.H0 += dmu * nbasis # ZHC NOTE FIXME add mu to H0?

    if "dmu_idx" in kwargs and kwargs["dmu_idx"] is not None:
        dmu_idx = kwargs["dmu_idx"]
    else:
        dmu_idx = range(lattice.nimp)

    if ImpHam.restricted:
        ImpHam.H1["cd"][0, dmu_idx, dmu_idx] -= dmu
    else:
        ImpHam.H1["cd"][0, dmu_idx, dmu_idx] -= dmu
        ImpHam.H1["cd"][1, dmu_idx, dmu_idx] -= dmu
    return ImpHam

def SolveImpHam_with_dmu(lattice, ImpHam, basis, dmu, solver, solver_args = {}, **kwargs):
    # H = H1 + Vcor - Mu
    # to keep H for mean-field Mu->Mu+dMu, Vcor->Vcor+dMu
    # In impurity Ham, equivalent to subtracting dMu from impurity, but not bath
    # The evaluation of energy is not affected if using (corrected) ImpHam-dMu
    # alternatively, we can change ImpHam.H0 to compensate
    ImpHam = apply_dmu(lattice, ImpHam, basis, dmu, **kwargs)
    result = solver.run(ImpHam, **solver_args)
    ImpHam = apply_dmu(lattice, ImpHam, basis, -dmu, **kwargs)
    return result

# FIXME it is better to define this class in a file contained under the folder routine/
class MuSolver(object):
    def __init__(self, adaptive=True, trust_region=2.5):
        self.adaptive = adaptive
        self.trust_region = trust_region
        self.history = []
        self.first_run = True

    def __call__(self, lattice, filling, ImpHam, basis, solver, \
            solver_args={}, delta=0.02, thrnelec=1e-5, step=0.05, \
            brentq_bound=None, brentq_value=None, nelec_sign=None, **kwargs):
        """
        Given impurity problems, fit mu and return solution.

        Returns:
            rhoEmb, EnergyEmb, ImpHam, dmu.
        """
        filling = np.average(filling) # for nelec_a and nelec_b
        single_imp = False
        if not isinstance(lattice, utils.Iterable):
            lattice = [lattice]
            ImpHam  = [ImpHam]
            basis   = [basis]
            solver  = [solver]
            solver_args = [solver_args]
            single_imp = True # only 1 impurity
        if nelec_sign is None:
            # default is sum up
            nelec_sign = [1 for i in range(len(lattice))]

        def solve_with_mu_loop(mu):
            """
            loop over all impurities and calculate nelec and sum up.
            """
            rhoEmb_col    = []
            EnergyEmb_col = []
            nelec_tot     = 0.0
            I = 0
            for lattice_I, ImpHam_I, basis_I, solver_I, solver_args_I in \
                    zip(lattice, ImpHam, basis, solver, solver_args):
                log.debug(0, "-" * 79)
                log.debug(0, "Solve impurity %s, indices: %s", I, \
                        utils.format_idx(lattice_I.imp_idx))
                rhoEmb_I, EnergyEmb_I = SolveImpHam_with_dmu(lattice_I, \
                        ImpHam_I, basis_I, mu, solver_I, solver_args_I, \
                        **kwargs)
                rhoEmb_col.append(rhoEmb_I)
                EnergyEmb_col.append(EnergyEmb_I)
                nelec = transformResults(rhoEmb_I, None, basis_I, None, \
                        None, lattice=lattice_I, **kwargs)
                nelec_tot += (nelec * nelec_sign[I])
                I += 1
            log.debug(0, "-" * 79)
            print("EnergyEmb_col", EnergyEmb_col)
            return rhoEmb_col, EnergyEmb_col, nelec_tot

        def apply_dmu_loop(dmu):
            """
            loop over all impurities and apply dmu.
            """
            ImpHam_col = []
            for lattice_I, ImpHam_I, basis_I in zip(lattice, ImpHam, basis):
                ImpHam_col.append(apply_dmu(lattice_I, ImpHam_I, basis_I, \
                        dmu, **kwargs))
            return ImpHam_col

        if brentq_bound is not None:
            from scipy.optimize import brentq
            lbound, rbound = brentq_bound
            def func(mu):
                log.info("mu (brentq guess): %s", mu)
                if brentq_value is not None:
                    if mu == lbound:
                        nelec = brentq_value[0]
                        log.info("mu : %s, dnelec : %s", mu, nelec-filling*2)
                        return nelec - filling * 2
                    elif mu == rbound:
                        nelec = brentq_value[1]
                        log.info("mu : %s, dnelec : %s", mu, nelec-filling*2)
                        return nelec - filling * 2

                nelec, rhoEmb, EnergyEmb = solve_with_mu_loop(mu)
                log.info("mu : %s, dnelec : %s", mu, nelec-filling*2)
                return nelec - filling*2

            delta = brentq(func, lbound, rbound, xtol=1e-5, rtol=1e-5, \
                    maxiter=20, full_output=False, disp=True)
            res = [rhoEmb, EnergyEmb, ImpHam, delta]
            if single_imp:
                res[0] = res[0][0]
                res[1] = res[1][0]
                res[2] = res[2][0]
            return res

        rhoEmb, EnergyEmb, nelec = solve_with_mu_loop(0.0)
        record = [(0.0, nelec)]
        log.result("nelec = %20.12f (target is %20.12f)", nelec, filling*2)
        #solver_args["similar"] = True

        if abs(nelec/(filling*2) - 1.0) < thrnelec:
            log.info("chemical potential fitting unnecessary")
            self.history.append(record)
            res = [rhoEmb, EnergyEmb, ImpHam, 0.0]
        else:
            if self.adaptive:
                # predict delta using historic information
                temp_delta = self.predict(nelec, filling*2)
                if temp_delta is not None:
                    delta = temp_delta
                    step = abs(delta) * self.trust_region
                else:
                    delta = abs(delta) * (-1 if (nelec > filling*2) else 1)
            else:
                delta = abs(delta) * (-1 if (nelec > filling*2) else 1)

            log.result("chemical potential fitting:\n" \
                    "finite difference dMu = %20.12f" % delta)
            rhoEmb1, EnergyEmb1, nelec1 = solve_with_mu_loop(delta)
            record.append((delta, nelec1))
            log.result("nelec = %20.12f (target is %20.12f)", nelec1, filling*2)

            if abs(nelec1/(filling*2) - 1.) < thrnelec:
                ImpHam = apply_dmu_loop(delta)
                self.history.append(record)
                res = [rhoEmb1, EnergyEmb1, ImpHam, delta]
            else:
                nprime = (nelec1 - nelec) / delta
                delta1 = (filling*2 - nelec) / nprime
                if abs(delta1) > step:
                    log.info("extrapolation dMu %20.12f more than trust step %20.12f", delta1, step)
                    delta1_tmp = copysign(step, delta1)
                    step = min(abs(delta1), 0.25)
                    delta1 = delta1_tmp
                log.info("dMu = %20.12f nelec = %20.12f", 0., nelec)
                log.info("dMu = %20.12f nelec = %20.12f", delta, nelec1)
                log.result("extrapolated to dMu = %20.12f", delta1)
                rhoEmb2, EnergyEmb2, nelec2 = solve_with_mu_loop(delta1)
                record.append((delta1, nelec2))
                log.result("nelec = %20.12f (target is %20.12f)", nelec2, filling*2)

                if abs(nelec2/(filling*2) - 1.) < thrnelec:
                    ImpHam = apply_dmu_loop(delta1)
                    self.history.append(record)
                    res = [rhoEmb2, EnergyEmb2, ImpHam, delta1]
                else:
                    mus = np.array([0.0, delta, delta1])
                    nelecs = np.array([nelec, nelec1, nelec2])
                    delta2 = quad_fit_mu(mus, nelecs, filling, step)

                    rhoEmb3, EnergyEmb3, nelec3 = solve_with_mu_loop(delta2)
                    record.append((delta2, nelec3))
                    log.result("nelec = %20.12f (target is %20.12f)", nelec3, filling * 2.0)

                    if abs(nelec3/(filling*2) - 1.) < thrnelec:
                        ImpHam = apply_dmu_loop(delta2)
                        self.history.append(record)
                        res = [rhoEmb3, EnergyEmb3, ImpHam, delta2]
                    else:
                        mus = np.array([0.0, delta, delta1, delta2])
                        nelecs = np.array([nelec, nelec1, nelec2, nelec3])
                        delta3 = quad_fit_mu(mus, nelecs, filling, step)

                        rhoEmb4, EnergyEmb4, nelec4 = solve_with_mu_loop(delta3)
                        record.append((delta3, nelec4))
                        log.result("nelec = %20.12f (target is %20.12f)", nelec4, filling * 2.0)

                        ImpHam = apply_dmu_loop(delta3)
                        self.history.append(record)
                        res = [rhoEmb4, EnergyEmb4, ImpHam, delta3]
        print(single_imp)
        if single_imp:
            res[0] = res[0][0]
            res[1] = res[1][0]
            res[2] = res[2][0]

        print("res", res)
        return res

    def save(self, filename):
        import pickle as p
        log.info("saving chemical potential fitting history to %s", filename)
        with open(filename, "wb") as f:
            p.dump(self.history, f)

    def load(self, filename):
        import pickle as p
        log.info("loading chemical potential fitting history from %s", filename)
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.history = p.load(f)
        else:
            log.warn("loading chemical potential fitting history fails.")

    def predict(self, nelec, target):
        # we assume the chemical potential landscape more or less the same for
        # previous fittings
        # the simplest thing to do is predicting a delta from each previous
        # fitting, and compute a weighted average. The weight should prefer
        # lattest runs, and prefer the fittigs that have has points close to
        # current and target filling
        from math import sqrt, exp
        vals = []
        weights = []

        # hyperparameters
        damp_factor = np.e
        sigma2, sigma3 = 0.00025, 0.0005

        for i, record in enumerate(self.history):
            # exponential
            weight = damp_factor ** (i + 1 - len(self.history))

            if len(record) == 1:
                val, weight = 0., 0.
                continue

            elif len(record) == 2:
                # we fit a line
                (mu1, n1), (mu2, n2) = record
                slope = (n2 - n1) / (mu2 - mu1)
                val = (target - nelec) / slope
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n2)**2 + (nelec-n1)**2)

                # Gaussian weight
                weight *= exp(- 0.5 * metric / sigma2)

            elif len(record) == 3:
                # we need to check data sanity: should be monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(record)
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0., 0.
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous
                a, b, c = np.dot(la.inv(np.asarray([
                    [mu1**2, mu1, 1],
                    [mu2**2, mu2, 1],
                    [mu3**2, mu3, 1]
                ])), np.asarray([n1,n2,n3]).reshape(-1,1)).reshape(-1)

                # if the parabola is not monotonic, use linear interpolation instead
                if mu1 < -0.5*b/a < mu3:
                    def find_mu(n):
                        if n < n2:
                            slope = (n2-n1) / (mu2-mu1)
                        else:
                            slope = (n2-n3) / (mu2-mu3)
                        return mu2 + (n-n2) / slope

                else:
                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n-n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n-n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c-n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n1)**2 + (nelec-n3)**2,
                        (target-n2)**2 + (nelec-n1)**2,
                        (target-n2)**2 + (nelec-n3)**2,
                        (target-n3)**2 + (nelec-n1)**2,
                        (target-n3)**2 + (nelec-n2)**2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            else: # len(record) >= 4:
                # first find three most nearest points
                mus, nelecs = zip(*record)
                mus = np.asarray(mus)
                nelecs = np.asarray(nelecs)
                delta_nelecs = np.abs(nelecs - target)
                idx_dN = np.argsort(delta_nelecs, kind='mergesort')
                mus_sub = mus[idx_dN][:3]
                nelecs_sub = nelecs[idx_dN][:3]


                # we need to check data sanity: should be monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(zip(mus_sub, nelecs_sub))
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0., 0.
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous
                a, b, c = np.dot(la.inv(np.asarray([
                    [mu1**2, mu1, 1],
                    [mu2**2, mu2, 1],
                    [mu3**2, mu3, 1]
                ])), np.asarray([n1,n2,n3]).reshape(-1,1)).reshape(-1)

                # if the parabola is not monotonic, use linear interpolation instead
                if mu1 < -0.5*b/a < mu3:
                    def find_mu(n):
                        if n < n2:
                            slope = (n2-n1) / (mu2-mu1)
                        else:
                            slope = (n2-n3) / (mu2-mu3)
                        return mu2 + (n-n2) / slope

                else:
                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n-n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n-n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c-n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n1)**2 + (nelec-n3)**2,
                        (target-n2)**2 + (nelec-n1)**2,
                        (target-n2)**2 + (nelec-n3)**2,
                        (target-n3)**2 + (nelec-n1)**2,
                        (target-n3)**2 + (nelec-n2)**2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            vals.append(val)
            weights.append(weight)

        log.debug(1, "dmu predictions:\n    value      weight")
        for v, w in zip(vals, weights):
            log.debug(1, "%10.6f %10.6f" % (v, w))

        if np.sum(weights) > 1e-3:
            dmu = np.dot(vals, weights) / np.sum(weights)
            if abs(dmu) > 0.5:
                dmu = copysign(0.5, dmu)
            log.info("adaptive chemical potential fitting, dmu = %20.12f", dmu)
            return dmu
        else:
            log.info("adaptive chemical potential fitting not used")
            return None

SolveImpHam_with_fitting = MuSolver(adaptive=True)

def AFInitGuess(ImpSize, U, Filling, polar=None, bogoliubov=False, rand=0.0, \
        subA=None, subB=None, bogo_res=False):
    """
    AFM initial guess.
    """
    if subA is None and subB is None:
        subA, subB = BipartiteSquare(ImpSize)
    nscsites = len(subA) + len(subB)
    shift = U * Filling
    if polar is None:
        polar = shift * Filling
    init_v = np.eye(nscsites) * shift
    init_p = np.diag([polar if s in subA else -polar for s in range(nscsites)])
    v = VcorLocal(False, bogoliubov, nscsites, bogo_res=bogo_res)
    if bogoliubov:
        np.random.seed(32499823)
        init_d = (np.random.rand(nscsites, nscsites) - 0.5) * rand
        v.assign(np.asarray([init_v+init_p, init_v-init_p, init_d]))
    else:
        v.assign(np.asarray([init_v+init_p, init_v-init_p]))
    return v

def PMInitGuess(ImpSize, U, Filling, bogoliubov=False, rand=0.0):
    """
    PM initial guess.
    """
    nscsites = np.product(ImpSize)
    shift = U * Filling
    init_v = np.eye(nscsites) * shift
    v = VcorLocal(True, bogoliubov, nscsites)
    if bogoliubov:
        init_d = np.zeros((nscsites, nscsites))
        v.assign(np.asarray([init_v, int_v, init_d]))
    else:
        v.assign(np.asarray([init_v, init_v]))

    if rand > 0.:
        np.random.seed(32499823)
        v.update(v.param + (np.random.rand(v.length()) - 0.5) * rand)
    return v

def VcorLocal(restricted, bogoliubov, nscsites, idx_range=None, bogo_res=False):
    """
    Local correlation potential.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        nscsites: number of orbitals.
        idx_range: index list to add correlation potential.
        bogo_res: if True, delta is restricted so that D = D.T

    Returns:
        vcor: vcor object.
    """
    if idx_range is None:
        idx_range = list(range(0, nscsites))
    nidx = len(idx_range)

    if restricted:
        nV = nidx * (nidx + 1) // 2
    else:
        nV = nidx * (nidx + 1)

    if bogoliubov and restricted:
        nD = nidx * (nidx + 1) // 2
    elif bogoliubov and not restricted:
        if bogo_res:
            nD = nidx * (nidx + 1) // 2
        else:
            nD = nidx * nidx
    else:
        nD = 0

    v = vcor.Vcor()
    v.restricted = restricted
    v.bogoliubov = bogoliubov
    v.bogo_res = bogo_res
    v.grad = None
    v.diag_idx = None

    if restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = 1
                self.grad = g
            return self.grad

        def diag_indices(self):
            if self.diag_idx is None:
                self.diag_idx = [utils.triu_diag_indices(len(idx_range))]
            return self.diag_idx

    elif not restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx + nV//2]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx+nV//2,1,i,j] = g[idx+nV//2,1,j,i] = 1
                self.grad = g
            return self.grad

        def diag_indices(self):
            if self.diag_idx is None:
                idx = utils.triu_diag_indices(len(idx_range))
                self.diag_idx = [idx, np.asarray(idx) + nV // 2]
            return self.diag_idx

    elif restricted and bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx]
                V[2,i,j] = V[2,j,i] = self.param[idx+nV]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = 1
                    g[idx+nV,2,i,j] = g[idx+nV,2,j,i] = 1
                self.grad = g
            return self.grad

        def diag_indices(self):
            if self.diag_idx is None:
                self.diag_idx = [utils.triu_diag_indices(len(idx_range))]
            return self.diag_idx

    else: # not restricted and bogoliubov
        if bogo_res:
            def evaluate(self):
                log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
                V = np.zeros((3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    V[0,i,j] = V[0,j,i] = self.param[idx]
                    V[1,i,j] = V[1,j,i] = self.param[idx+nV//2]
                    V[2,i,j] = V[2,j,i] = self.param[idx+nV]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nV+nD, 3, nscsites, nscsites))
                    for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                        g[idx,0,i,j] = g[idx,0,j,i] = 1
                        g[idx+nV//2,1,i,j] = g[idx+nV//2,1,j,i] = 1
                        g[idx+nV,2,i,j] = g[idx+nV,2,j,i] = 1
                    self.grad = g
                return self.grad
        else:
            def evaluate(self):
                log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
                V = np.zeros((3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    V[0,i,j] = V[0,j,i] = self.param[idx]
                    V[1,i,j] = V[1,j,i] = self.param[idx+nV//2]
                for idx, (i,j) in enumerate(it.product(idx_range, repeat = 2)):
                    V[2,i,j] = self.param[idx+nV]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nV+nD, 3, nscsites, nscsites))
                    for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                        g[idx,0,i,j] = g[idx,0,j,i] = 1
                        g[idx+nV//2,1,i,j] = g[idx+nV//2,1,j,i] = 1
                    for idx, (i,j) in enumerate(it.product(idx_range, repeat = 2)):
                        g[idx+nV,2,i,j] = 1
                    self.grad = g
                return self.grad

        def diag_indices(self):
            if self.diag_idx is None:
                idx = utils.triu_diag_indices(len(idx_range))
                self.diag_idx = [idx, np.asarray(idx) + nV // 2]
            return self.diag_idx

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.diag_indices = types.MethodType(diag_indices, v)
    v.length = types.MethodType(lambda self: nV+nD, v)
    v.idx_range = idx_range
    return v

def vcor_zeros(restricted, bogoliubov, nscsites, idx_range=None, \
        bogo_res=False):
    """
    Initialize elements of vcor as zeros.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        nscsites: number of orbitals.
        idx_range: index list to add correlation potential.
        bogo_res: if True, delta is restricted so that D = D.T

    Returns:
        vcor: vcor object with all elements are zero.
    """
    vcor = VcorLocal(restricted, bogoliubov, nscsites, idx_range=idx_range, \
            bogo_res=bogo_res)
    vcor.update(np.zeros(vcor.length()))
    return vcor

def VcorRestricted(restricted, bogoliubov, active_sites, core_sites, \
        bogo_res=False, nscsites=None):
    """
    Full correlation potential for active sites,
    diagonal potential for core sites.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        active_sites: list of active orbital indices.
        core_sites: list of core orbital indices.
        bogo_res: if True, delta is restricted so that D = D.T
        nscsites: optional, if None, will be set to nact + ncore

    Returns:
        vcor: vcor object.
    """
    nAct = len(active_sites)
    nCor = len(core_sites)
    if nscsites is None:
        nscsites = nAct + nCor
    else:
        if nscsites != nAct + nCor:
            log.warn("nscsites (%s) != nAct (%s) + nCor (%s)", nscsites, \
                    nAct, nCor)

    if restricted:
        nV0 = nAct * (nAct + 1) // 2
        nV = nV0 + nCor
    else:
        nV0 = nAct * (nAct + 1)
        nV = nV0 + nCor * 2

    # no bogoliubov term on core sites
    if bogoliubov and restricted:
        nD = nAct * (nAct + 1) // 2
    elif bogoliubov and not restricted:
        if bogo_res:
            nD = nAct * (nAct + 1) // 2
        else:
            nD = nAct * nAct
    else:
        nD = 0

    v = vcor.Vcor()
    v.restricted = restricted
    v.bogoliubov = bogoliubov
    v.bogo_res = bogo_res
    v.grad = None

    if restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((1, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = self.param[nV0 + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 1, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = 1
                self.grad = g
            return self.grad

    elif not restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
                V[1, i, j] = V[1, j, i] = self.param[nV0//2 + idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = self.param[nV0 + idx]
                V[1, i, i] = self.param[nV0 + nCor + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                    g[nV0//2 + idx, 1, i, j] = g[nV0//2 + idx, 1, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = 1
                    g[nV0 + nCor + idx, 1, i, i] = 1
                self.grad = g
            return self.grad

    elif restricted and bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
                V[1, i, j] = V[1, j, i] = self.param[idx]
                V[2, i, j] = V[2, j, i] = self.param[nV + idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = V[1, i, i] = self.param[nV0 + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                    g[idx, 1, i, j] = g[idx, 1, j, i] = 1
                    g[nV + idx, 2, i, j] = g[nV + idx, 2, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = g[nV0 + idx, 1, i, i] = 1
                self.grad = g
            return self.grad

    else:
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                V[0, i, j] = V[0, j, i] = self.param[idx]
                V[1, i, j] = V[1, j, i] = self.param[nV0//2 + idx]
            for idx, i in enumerate(core_sites):
                V[0, i, i] = self.param[nV0 + idx]
                V[1, i, i] = self.param[nV0 + nCor + idx]
            for idx, (i, j) in enumerate(it.product(active_sites, repeat = 2)):
                V[2, i, j] = self.param[nV + idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i, j) in enumerate(it.combinations_with_replacement(active_sites, 2)):
                    g[idx, 0, i, j] = g[idx, 0, j, i] = 1
                    g[nV0//2 + idx, 1, i, j] = g[nV0//2 + idx, 1, j, i] = 1
                for idx, i in enumerate(core_sites):
                    g[nV0 + idx, 0, i, i] = 1
                    g[nV0 + nCor + idx, 1, i, i] = 1
                for idx, (i, j) in enumerate(it.product(active_sites, repeat = 2)):
                    g[nV + idx, 2, i, j] = 1
                self.grad = g
            return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nV + nD, v)
    return v

def VcorSymm(restricted, bogoliubov, nscsites, C_symm, idx_range=None, \
        bogo_res=False):
    """
    Local correlation potential. Point group symmetry.

    Args:
        restricted: restricted on v.
        bogoliubov: whether include delta.
        nscsites: number of orbitals.
        idx_range: index list to add correlation potential.
        bogo_res: if True, delta is restricted so that D = D.T

    Returns:
        vcor: vcor object.
    """
    # param will be (nirep, spin, nV+nD)

    if idx_range is None:
        idx_range = list(range(0, nscsites))
    idx_mesh = np.ix_(idx_range, idx_range)

    nirep = len(C_symm)
    norb_list = []
    nV_list   = []
    nD_list   = []
    for C in C_symm:
        nidx = C.shape[-1]

        if restricted:
            nV = nidx * (nidx + 1) // 2
        else:
            nV = nidx * (nidx + 1)

        if bogoliubov and restricted:
            nD = nidx * (nidx + 1) // 2
        elif bogoliubov and not restricted:
            if bogo_res:
                nD = nidx * (nidx + 1) // 2
            else:
                nD = nidx * nidx
        else:
            nD = 0

        norb_list.append(nidx)
        nV_list.append(nV)
        nD_list.append(nD)

    assert np.sum(norb_list) == C_symm[0].shape[0]
    assert len(idx_range) == C_symm[0].shape[0]
    nparam_tot = np.sum(nV_list) + np.sum(nD_list)

    v = vcor.Vcor()
    v.restricted = restricted
    v.bogoliubov = bogoliubov
    v.bogo_res = bogo_res
    v.grad = None
    v.diag_idx = None

    if restricted and not bogoliubov:
        raise NotImplementedError
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = 1
                self.grad = g
            return self.grad

    elif not restricted and not bogoliubov:
        def evaluate(self):
            log.eassert(self.param.shape == (nparam_tot,), \
                    "wrong parameter shape, require %s", (nparam_tot,))
            V = np.zeros((2, nscsites, nscsites))
            stepi = 0
            for irep, C in enumerate(C_symm):
                nstep = nV_list[irep] // 2

                stepf = stepi + nstep
                V[0][idx_mesh] += mdot(C, lib.unpack_tril(self.param[stepi:stepf]), C.conj().T)
                stepi = stepf

                stepf = stepi + nstep
                V[1][idx_mesh] += mdot(C, lib.unpack_tril(self.param[stepi:stepf]), C.conj().T)
                stepi = stepf
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nparam_tot, 2, nscsites, nscsites))
                stepi = 0
                for irep, C in enumerate(C_symm):
                    nstep = nV_list[irep] // 2
                    tmp = np.zeros((nstep,))

                    stepf = stepi + nstep
                    for i in range(nstep):
                        tmp[:] = 0.0
                        tmp[i] = 1.0
                        g[stepi+i, 0][idx_mesh] = mdot(C, lib.unpack_tril(tmp), C.conj().T)
                    stepi = stepf

                    stepf = stepi + nstep
                    for i in range(nstep):
                        tmp[:] = 0.0
                        tmp[i] = 1.0
                        g[stepi+i, 1][idx_mesh] = mdot(C, lib.unpack_tril(tmp), C.conj().T)
                    stepi = stepf
                self.grad = g
            return self.grad

        def diag_indices(self):
            if self.diag_idx is None:
                idx_a = []
                idx_b = []
                offset = 0
                for irep, C in enumerate(C_symm):
                    nmo = C.shape[-1]
                    idx = utils.tril_diag_indices(nmo) + offset
                    tmp = nmo * (nmo + 1) // 2
                    idx_a.extend(idx)
                    idx_b.extend(idx + tmp)
                    offset += (tmp * 2)
                self.diag_idx = [idx_a, idx_b]
            return self.diag_idx

    elif restricted and bogoliubov:
        raise NotImplementedError
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = self.param[idx]
                V[2,i,j] = V[2,j,i] = self.param[idx+nV]
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = 1
                    g[idx+nV,2,i,j] = g[idx+nV,2,j,i] = 1
                self.grad = g
            return self.grad

    else: # not restricted and bogoliubov
        raise NotImplementedError
        if bogo_res:
            def evaluate(self):
                log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
                V = np.zeros((3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    V[0,i,j] = V[0,j,i] = self.param[idx]
                    V[1,i,j] = V[1,j,i] = self.param[idx+nV//2]
                    V[2,i,j] = V[2,j,i] = self.param[idx+nV]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nV+nD, 3, nscsites, nscsites))
                    for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                        g[idx,0,i,j] = g[idx,0,j,i] = 1
                        g[idx+nV//2,1,i,j] = g[idx+nV//2,1,j,i] = 1
                        g[idx+nV,2,i,j] = g[idx+nV,2,j,i] = 1
                    self.grad = g
                return self.grad
        else:
            def evaluate(self):
                log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
                V = np.zeros((3, nscsites, nscsites))
                for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                    V[0,i,j] = V[0,j,i] = self.param[idx]
                    V[1,i,j] = V[1,j,i] = self.param[idx+nV//2]
                for idx, (i,j) in enumerate(it.product(idx_range, repeat = 2)):
                    V[2,i,j] = self.param[idx+nV]
                return V

            def gradient(self):
                if self.grad is None:
                    g = np.zeros((nV+nD, 3, nscsites, nscsites))
                    for idx, (i,j) in enumerate(it.combinations_with_replacement(idx_range, 2)):
                        g[idx,0,i,j] = g[idx,0,j,i] = 1
                        g[idx+nV//2,1,i,j] = g[idx+nV//2,1,j,i] = 1
                    for idx, (i,j) in enumerate(it.product(idx_range, repeat = 2)):
                        g[idx+nV,2,i,j] = 1
                    self.grad = g
                return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.diag_indices = types.MethodType(diag_indices, v)
    v.length = types.MethodType(lambda self: nparam_tot, v)
    v.idx_range = idx_range
    return v

VcorLocal_new = VcorLocal

VcorZeros = vcor_zeros

VcorNonLocal = vcor.VcorNonLocal

addDiag = slater.addDiag

make_vcor_trace_unchanged = slater.make_vcor_trace_unchanged

FitVcor = slater.FitVcorTwoStep

unit2emb = slater.unit2emb

