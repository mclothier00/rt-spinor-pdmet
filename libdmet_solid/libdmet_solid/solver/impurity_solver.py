from libdmet_solid.solver import block
from libdmet_solid.solver.afqmc import AFQMC
from libdmet_solid.system import integral
from libdmet_solid.utils import logger as log
import numpy as np
import scipy.linalg as la

from libdmet_solid.solver import scf, casscf, bcs_dmrgscf
from libdmet_solid.solver.dmrgci import CASCI, DmrgCI, get_orbs
from libdmet_solid.solver.bcs_dmrgci import BCSDmrgCI, get_qps, get_BCS_mo
from libdmet_solid.solver.fci import FCI, FCI_AO
from libdmet_solid.solver.cc import CCSD
from libdmet_solid.solver.scf_solver import SCFSolver

__all__ = ["AFQMC", "Block", "StackBlock", "DmrgCI", "CASSCF", "BCSDmrgCI",\
        "FCI", "FCI_AO", "CCSD", "SCFSolver"]
try:
    from libdmet_solid.solver.shci import SHCI
    __all__.append("SHCI")
except ImportError:
    log.info("ImportError in SHCI solver, settings.py should be set in pyscf")

class Block(object):
    def __init__(self, nproc, nnode=1, TmpDir="./tmp", SharedDir=None, \
            reorder=False, minM=100, maxM=None, tol=1e-6, spinAdapted=False, \
            bcs=False, ghf=False, maxiter_initial=28, maxiter_restart=14):
        log.eassert(nnode == 1 or SharedDir is not None, \
                "Running on multiple nodes (nnod = %d), must specify shared directory", \
                nnode)
        self.cisolver = block.Block()
        block.Block.set_nproc(nproc, nnode)
        self.cisolver.createTmp(tmp = TmpDir, shared = SharedDir)
        block.Block.reorder = reorder
        self.schedule = block.Schedule(sweeptol = tol)
        if minM > maxM:
            minM = maxM
        self.minM = minM
        self.maxM = maxM
        self.spinAdapted = spinAdapted
        self.bcs = bcs
        self.ghf = ghf
        self.maxiter_initial = maxiter_initial
        self.maxiter_restart = maxiter_restart

    def run(self, Ham, M=None, nelec=None, schedule=None, similar=False, \
            restart=True, spin=0, **kwargs):
        if M is None:
            M = self.maxM
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                nelec = Ham.norb
        if not self.cisolver.sys_initialized:
            self.cisolver.set_system(nelec, spin, self.spinAdapted, \
                    self.bcs, (self.spinAdapted and (not self.ghf)))

        if schedule is None:
            schedule = self.schedule
            if self.cisolver.optimized and restart:
                schedule.maxiter = self.maxiter_restart
                schedule.gen_restart(M)
            else:
                self.cisolver.optimized = False
                schedule.maxiter = self.maxiter_initial
                schedule.gen_initial(minM = self.minM, maxM = M)

        self.cisolver.set_schedule(schedule)
        self.cisolver.set_integral(Ham)

        truncation, energy, onepdm = self.cisolver.optimize()
        return onepdm, energy
    
    def run_dmet_ham(self, Ham, M = None, **kwargs):
        #if M is None:
        #    M = self.maxM
        #log.eassert(self.cisolver.sys_initialized, \
        #        "cisolver should be initialized for fixed wave")
        
        #schedule = self.schedule
        #schedule.maxiter = 1
        #schedule.gen_fixwave(M)
        
        #self.cisolver.set_schedule(schedule)
        #self.cisolver.set_integral(Ham)
        #self.cisolver.fixwave = True # to write OH.conf and wavenum

        # ZHC NOTE FIXME
        # use another fold to run?
        #truncation, energy, onepdm = self.cisolver.optimize()
        E = self.cisolver.evaluate(Ham.H0, Ham.H1, Ham.H2, op = "dmet ham")
        return E

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.cisolver.onepdm()

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.cisolver.twopdm()

    def cleanup(self):
        # FIXME first copy and save restart files
        self.cisolver.cleanup()

class StackBlock(Block):
    def __init__(self, nproc, nthread=1, nnode=1, TmpDir="./tmp", \
            SharedDir=None, reorder=False, minM=100, maxM=None, tol=1e-6, \
            spinAdapted=False, bcs=False, ghf=False, mem=80, \
            maxiter_initial=28, maxiter_restart=14):
        log.eassert(nnode == 1 or SharedDir is not None, \
                "Running on multiple nodes (nnod = %d)," \
                "must specify shared directory", nnode)
        self.cisolver = block.StackBlock()
        block.StackBlock.set_nproc(nproc, nthread, nnode)
        self.cisolver.createTmp(tmp = TmpDir, shared = SharedDir)
        block.StackBlock.reorder = reorder
        self.schedule = block.Schedule(sweeptol = tol)
        if minM > maxM:
            minM = maxM
        self.minM = minM
        self.maxM = maxM
        self.spinAdapted = spinAdapted
        self.bcs = bcs
        self.ghf = ghf
        self.cisolver.mem = mem
        self.maxiter_initial = maxiter_initial
        self.maxiter_restart = maxiter_restart

class CASSCF(object):

    settings = {
        "max_stepsize": 0.04,
        "max_cycle_macro": 50,
        "max_cycle_micro": 8, # micro_cycle
        "max_cycle_micro_inner": 10,
        "conv_tol": 1e-5, # energy convergence
        "conv_tol_grad": None, # orb grad convergence
        # for augmented hessian
        "ah_level_shift": 1e-4,
        "ah_conv_tol": 1e-12, # augmented hessian accuracy
        "ah_max_cycle": 30,
        "ah_lindep": 1e-14,
        "ah_start_tol": 0.1,
        "ah_start_cycle": 2,
        "ah_grad_trust_region": 4,
        "ah_guess_space": 0,
        "ah_decay_rate": 0.7, # augmented hessian decay
        "ci_repsonse_space": 3,
        "keyframe_interval": 1000000, # do not change
        "keyframe_trust_region": 0,  # do not change
        "dynamic_micro_step": False,
        "exact_integral": True,
    }

    def __init__(self, ncas, nelecas = None, bogoliubov = False, \
            MP2natorb = False, spinAverage = False, fcisolver = "FCI", \
            settings = {}):
        log.eassert(ncas * 2 >= nelecas, \
                "CAS size not compatible with number of electrons")
        self.ncas = ncas
        self.nelecas = nelecas # alpha and beta
        self.MP2natorb = MP2natorb
        self.spinAverage = spinAverage
        self.bogoliubov = bogoliubov
        self.scfsolver = scf.SCF()
        self.mo_coef = None

        # mcscf options, these will be used for first installation of
        # mcscf class: casscf.CASSCF or casscf.DMRGSCF
        self.settings = settings
        if fcisolver.upper() == "FCI":
            log.eassert(not bogoliubov, \
                    "FCI solver is not available for BCS calculations")
            self.solver_cls = casscf.CASSCF
        elif fcisolver.upper() == "DMRG":
            log.eassert(self.settings.has_key("fcisolver"), \
                    "When using DMRG-CASSCF, must specify "
                    "the key 'fcisolver' with a BLOCK solver instance")
            if bogoliubov:
                self.solver_cls = bcs_dmrgscf.BCS_DMRGSCF
            else:
                self.solver_cls = casscf.DMRGSCF
        else:
            log.error("FCI solver %s is not valid.", fcisolver.upper())
        # solver instance, initialized at first run
        self.solver = None

    def apply_options(self, mc, options):
        for key, val in options.items():
            setattr(mc, key, val)

    def run(self, Ham, mcscf_args = {}, guess = None, nelec = None, \
            similar = False):
        if self.bogoliubov:
            return self._run_bogoliubov(Ham, mcscf_args, guess, similar)

        spin = Ham.H1["cd"].shape[0]
        norbs = Ham.H1["cd"].shape[1]
        if nelec is None:
            nelec = Ham.norb
        log.eassert(spin == 2, \
                "spin-restricted CASSCF solver is not implemented")

        nelecasAB = (self.nelecas//2, self.nelecas//2)

        if self.mo_coef is None or not similar: 
            # not restart from previous orbitals
            log.debug(0, "Generate new orbitals using Hartree-Fock")
            core, cas, virt, _ = get_orbs(self, Ham, guess, nelec)
            self.mo_coef = np.empty((2, norbs, norbs))
            self.mo_coef[:, :, :core.shape[2]] = core
            self.mo_coef[:, :, core.shape[2]:core.shape[2]+cas.shape[2]] = cas
            self.mo_coef[:, :, core.shape[2]+cas.shape[2]:] = virt
        else:
            log.debug(0, "Reusing previous orbitals")
            spin = Ham.H1["cd"].shape[0]
            self.scfsolver.set_system(nelec, 0, False, spin == 1)
            self.scfsolver.set_integral(Ham)
            self.scfsolver.HF(tol = 1e-5, MaxIter = 30, InitGuess = guess)

        if self.solver is None:
            self.solver = self.solver_cls(self.scfsolver.mf, \
                    self.ncas, nelecasAB, **self.settings)
        else:
            self.solver.refresh(self.scfsolver.mf, self.ncas, nelecasAB)

        # apply options, these options are limited to the CASSCF convergence
        # settings specified as static members
        self.apply_options(self.solver, CASSCF.settings)

        E, _, _, self.mo_coef = self.solver.mc1step(mo_coeff = self.mo_coef, \
                **mcscf_args)
        rho = np.asarray(self.solver.make_rdm1s())

        return rho, E

    def _run_bogoliubov(self, Ham, mcscf_args = {}, guess = None, similar = False):
        spin = 2
        norbs = Ham.H1["cd"].shape[1]

        if self.nelecas is None:
            fget_qps = get_qps(self.ncas, algo = "energy")
        else:
            fget_qps = get_qps(self.ncas, self.nelecas, algo = "nelec")
        # I don't use local algo here

        if self.mo_coef is None or not similar:
            # not restart from previous orbitals
            log.debug(0, "Generate new quasiparticles using HFB")
            mo, mo_energy = get_BCS_mo(self.scfsolver, Ham, guess)
            core, cas, _ = fget_qps(mo, mo_energy)
            self.mo_coef = np.empty((2, norbs*2, norbs))
            self.mo_coef[:, :, :core.shape[2]] = core
            self.mo_coef[:, :, core.shape[2]:] = cas
        else:
            log.debug(0, "Reusing previous quasiparticles")
            self.scfsolver.set_system(None, 0, True, False)
            self.scfsolver.set_integral(Ham)
            self.scfsolver.HFB(Mu = 0, tol = 1e-5, MaxIter = 30, \
                    InitGuess = guess)

        if self.solver is None:
            self.solver = self.solver_cls(self.scfsolver.mf, \
                    self.ncas, norbs, Ham, nelecas = self.nelecas, \
                    **self.settings)
        else:
            self.solver.refresh(self.scfsolver.mf, self.ncas, \
                    norbs, Ham, nelecas = self.nelecas)

        self.apply_options(self.solver, CASSCF.settings)
        E, _, _, self.mo_coef = self.solver.mc1step(mo_coeff = self.mo_coef, \
                **mcscf_args)

        from libdmet_solid.routine.bcs_helper import combineRdm
        rho, kappa = self.solver.make_rdm1s()
        GRho = combineRdm(rho[0], rho[1], kappa)

        return GRho, E

    def cleanup(self):
        pass


