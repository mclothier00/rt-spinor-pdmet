# Routines to run a real-time projected-DMET calculation
import numpy as np
import sys
import real_time_pDMET.rtpdmet.dynamics.mf1rdm_timedep_mod as mf1rdm_timedep_mod
import real_time_pDMET.scripts.applyham_pyscf as applyham_pyscf
import real_time_pDMET.scripts.make_hams as make_hams 
import real_time_pDMET.scripts.utils as utils
import pickle
import time
import math
# ########### CLASS TO RUN REAL-TIME DMET CALCULATION #########

class dynamics_driver():

    #####################################################################
    def __init__(self, h_site, V_site, hamtype, tot_system, delt, dG,
                 dX, U, A_nott, t_nott, omega, t_d, nproc, Nstep,
                 Nprint=100, integ='rk1', hubsite_indx=None,
                 laser_sites=None, init_time=0.0, laser=False):

        # h_site -
        # 1 e- hamiltonian in site-basis for total system to run dynamics
        # V_site -
        # 2 e- hamiltonian in site-basis for total system to run dynamics
        # hamtype -
        # integer defining if using a special Hamiltonian (Hubbard or Anderson)
        # tot_system -
        # previously defined DMET total system including fragment information
        # delt - time step
        # Nstep - total number of time-steps
        # Nprint - number of time-steps between printing
        # init_time - the starting time for the calculation
        # integ - the type of integrator used
        # nproc - number of processors for calculation
        # - careful, there is no check that this matches the pbs script

        self.tot_system = tot_system
        self.delt = delt
        self.Nstep = Nstep
        self.Nprint = Nprint
        self.init_time = init_time
        self.integ = integ
        self.nproc = nproc
        self.dG = dG
        self.dX = dX
        self.U = U
        self.A_nott = A_nott
        self.t_nott = t_nott
        self.t_d = t_d
        self.omega = omega
        self.laser = laser
        self.laser_sites = laser_sites
        if not (np.diagonal(h_site)).any() == True:
            self.Vbias = True
        else:
            self.Vbias = False

        print()
        print('********************************************')
        print('     SET-UP REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()
        # Input error checks
        '''
         requiers care, do not use if t=0 is a highly degenerate
         space, can result in generation of  non-continuous
         locally degenerate eigenvalues during diagonalization
        '''
        # self.import_check(1e-7)

        # Convert rotation matrices, CI coefficients,
        # and MF 1RDM to complex arrays if they're not already
        for frag in self.tot_system.frag_list:
            if(not np.iscomplexobj(frag.rotmat)):
                frag.rotmat = frag.rotmat.astype(complex)
            if(not np.iscomplexobj(frag.CIcoeffs)):
                frag.CIcoeffs = frag.CIcoeffs.astype(complex)

        if(not np.iscomplexobj(self.tot_system.mf1RDM)):
            self.tot_system.mf1RDM = self.tot_system.mf1RDM.astype(complex)

        if(not np.iscomplexobj(self.tot_system.glob1RDM)):
            self.tot_system.glob1RDM = self.tot_system.glob1RDM.astype(complex)

        if(not np.iscomplexobj(self.tot_system.NOevecs)):
            self.tot_system.NOevecs = self.tot_system.NOevecs.astype(complex)

        if self.laser==False:
            self.tot_system.h_site = np.real(h_site)
        else:
            if(not np.iscomplexobj(self.tot_system.h_site)):
                self.tot_system.h_site = self.tot_system.h_site.astype(complex)

        # Set-up Hamiltonian for dynamics calculation
        self.tot_system.V_site = V_site
        self.tot_system.hamtype = hamtype

        # If running Hubbard-like model, need an array
        # containing index of all sites that have hubbard U term
        self.tot_system.hubsite_indx = hubsite_indx
        if(self.tot_system.hamtype == 1
           and self.tot_system.hubsite_indx is None):
            print('ERROR: Did not specify an array of sites that have U term')
            print()
            exit()

        # Define output files
        self.file_output = open('output_dynamics.dat', 'w')
        self.file_corrdens = open('electron_density.dat', 'w')
        if self.laser == True:
            self.file_laser = open('laser.dat', 'w')
        if self.Vbias == True:
            self.file_current = open('current.dat', 'w')

        self.max_diagonalG = 0
        self.corrdens_old = np.zeros((self.tot_system.Nsites))
        # self.corrdens_old += 1

        # Paralelization is under development

        # start_pool = time.time()
        # self.frag_pool = multproc.Pool(nproc)
        # print("time to from pool", time.time()-start_pool)
 
 #####################################################################
    def kernel(self):
        start_time = time.time()
        print()
        print('********************************************')
        print('     BEGIN REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()

        # fraddg_pool = multproc.Pool(self.nproc)

        # DYNAMICS LOOP
        current_time = self.init_time

        for step in range(self.Nstep):
            # Print data
            self.step = step
            if step == 0:
                self.print_just_dens(current_time)
                sys.stdout.flush()

            if(np.mod(step, self.Nprint) == 0) and step > 1:
                print('Writing data at step ', step, 'and time',
                      current_time, 'for RT-pDMET calculation')
                self.print_data(current_time)
                sys.stdout.flush()

            # if a trajectory restarted, record data before a 1st step
            if current_time != 0 and step == 0:
                print('Writing data at step ', step, 'and time',
                      current_time, 'for RT-pDMET calculation')
                self.print_data(current_time)
                sys.stdout.flush()

            # Integrate FCI coefficients and rotation matrix for all fragments
            self.integrate(self.nproc, current_time)

            # Increase current_time
            current_time = self.init_time + (step+1)*self.delt
            sys.stdout.flush()

        # Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time',
              current_time, 'for RT-pDMET calculation')
        self.print_data(current_time)
        sys.stdout.flush()

        # Close output files
        self.file_output.close()
        self.file_corrdens.close()

        if self.Vbias == True:
            self.file_current.close()
        if self.laser == True:
            self.file_laser.close()

        # self.frag_pool.close()
        print()
        print('********************************************')
        print('       END REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()
        print("--- %s seconds ---" % (time.time() - start_time))
 
 #####################################################################
    def update_ham(self, curr_time):
        ## NOTE: need to think about how laser interacts with spin states; current
        ## equations do not take this into account
        laser_pulse =(
            self.A_nott*np.exp(-((curr_time-self.t_nott)**2)/(2*self.t_d**2))
            *math.cos(self.omega*(curr_time-self.t_nott)))
        Nimp = np.shape(self.tot_system.hubsite_indx)[0]
        self.tot_system.h_site =(
            make_hams.make_ham_multi_imp_anderson_realspace_laser(
                self.tot_system.Nsites, self.U, laser_pulse, self.laser_sites,
                self.tot_system.hubsite_indx, t=1, update=True,
                boundary=False, Full=False))

        fmt_str = '%20.8e'
        laser_time = np.array([curr_time, laser_pulse])
        np.savetxt( self.file_laser, laser_time.reshape(1,2), fmt_str)
        self.file_laser.flush()

#####################################################################
    def integrate(self, nproc, current_time):
        # Subroutine to integrate equations of motion
        if(self.integ == 'rk4'):

            # Use 4th order runge-kutta to integrate EOMs

            # Copy MF 1RDM, global RDM, CI coefficients,
            # natural and embedding orbs at time t
            init_NOevecs = np.copy(self.tot_system.NOevecs)
            init_glob1RDM = np.copy(self.tot_system.glob1RDM)
            init_mf1RDM = np.copy(self.tot_system.mf1RDM)

            init_CIcoeffs_list = []
            init_rotmat_list = []
            for frag in self.tot_system.frag_list:
                init_rotmat_list.append(np.copy(frag.rotmat))
                init_CIcoeffs_list.append(np.copy(frag.CIcoeffs))

            # Calculate appropriate changes in MF 1RDM,
            # embedding orbitals, and CI coefficients
            # l, k, m, n, p =
            # change in NOevecs, rotmat_list, CIcoeffs_list, glob1RDM, mf1RDM
            
            # GETTING 1ST SUBSTEP DT

            l1, k1_list, m1_list, n1, p1, mfRDM_check = self.one_rk_step(nproc)
            
            self.tot_system.NOevecs = init_NOevecs + 0.5*l1
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n1
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p1
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = init_rotmat_list[cnt] + 0.5*k1_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m1_list[cnt]

            if self.laser==True:
                self.update_ham(current_time+0.5*self.delt)

            # GETTING 2ST SUBSTEP DT

            l2, k2_list, m2_list, n2, p2, mfRDM_check = self.one_rk_step(nproc)

            self.tot_system.NOevecs = init_NOevecs + 0.5*l2
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n2
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p2
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = init_rotmat_list[cnt] + 0.5*k2_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]

            if self.laser==True:
                self.update_ham(current_time+0.5*self.delt)

            # GETTING 3ST SUBSTEP DT

            l3, k3_list, m3_list, n3, p3, mfRDM_check = self.one_rk_step(nproc)

            self.tot_system.NOevecs = init_NOevecs + 1.0*l3
            self.tot_system.glob1RDM = init_glob1RDM + 1.0*n3
            self.tot_system.mf1RDM = init_mf1RDM + 1.0*p3
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = init_rotmat_list[cnt] + 1.0*k3_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*m3_list[cnt]

            if self.laser==True:
                self.update_ham(current_time+1.0*self.delt)

            # GETTING 4ST SUBSTEP DT

            l4, k4_list, m4_list, n4, p4, mfRDM_check = self.one_rk_step(nproc)

            self.tot_system.NOevecs = (
                init_NOevecs + 1.0/6.0 * (l1 + 2.0*l2 + 2.0*l3 + l4))
            self.tot_system.glob1RDM = (
                init_glob1RDM + 1.0/6.0 * (n1 + 2.0*n2 + 2.0*n3 + n4))
            self.tot_system.mf1RDM = (
                init_mf1RDM + 1.0/6.0 * (p1 + 2.0*p2 + 2.0*p3 + p4))
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = (
                    init_rotmat_list[cnt] + 1.0/6.0 * (
                        k1_list[cnt] + 2.0*k2_list[cnt] +
                        2.0*k3_list[cnt] + k4_list[cnt]))

                frag.CIcoeffs = (
                    init_CIcoeffs_list[cnt] + 1.0/6.0 * (
                        m1_list[cnt] + 2.0*m2_list[cnt] +
                        2.0*m3_list[cnt] + m4_list[cnt]))

            if self.laser==True:
                self.update_ham(current_time+1.0*self.delt)

            # Checks for numerical stability 
            
            # check whether two analitically equivalent methods to form
            # ddt mfRDM are numerically equivalent 
            
            if self.step==0:
                self.mfRDM_check_old = mfRDM_check

            if mfRDM_check==False and self.mfRDM_check_old==True:
                print(" ")
                text_warn='''ddt of mean-field 1RDM formed from natural orbitals
                            is different from ddt of mean-field 1RDM
                            formed as a commutator of G and mf 1RDM'''
                text_centered = text_warn.center(24)
                print("NUMERICAL STABILITY WARNING:", text_centered)
                print()
                print("discrepancy threshold dG=",self.dG, "at time", current_time)
                print()

            self.mfRDM_check_old = mfRDM_check
            
            # check how well natural orbitals diagonalize global rdm
            
            diag_global = (utils.rot1el(
                self.tot_system.glob1RDM, self.tot_system.NOevecs))
            np.fill_diagonal(diag_global, 0)
            self.max_diag_global = self.return_max_value(diag_global)
            
            diag_globalRDM_check = (
                    np.allclose(diag_global, 
                                np.zeros((self.tot_system.Nsites, self.tot_system.Nsites)), 
                                rtol=0, atol=self.dG))
            if self.step==0:
                self.diag_globalRDM_check_old = diag_globalRDM_check

            if diag_globalRDM_check==False and self.diag_globalRDM_check_old==True:
                print(" ")
                text_warn='''missmatch between global 1RDM and its propagated egenvectors;
                            off-diagonal terms of global 1RDM diagonalized with NOs are 
                            greater than a tollerance threshold dG'''
                text_centered = text_warn.center(24)
                print("NUMERICAL STABILITY WARNING:", text_centered)
                print()
                print("discrepancy threshold dG=",self.dG, "at time", current_time)
                print()
            
            self.diag_globalRDM_check_old = diag_globalRDM_check 
            if np.isnan(np.sum(diag_global))==True:
                text_warn = '''Dynamics diverged due to the missmatch between NOs and global 1RDM; 
                            this is associated with degeneracies between global 1RDM eigenvalues; 
                            consider increasing dG threshold;
                            QUITTING THE SIMULATION'''
                text_centered = text_warn.center(24)
                print("ERROR", text_centered)
                quit()
        else:
            print('ERROR: A proper integrator was not specified')
            exit()

            quit()
    #####################################################################
    def return_max_value(self, array):
        largest = 0
        for x in range(0, len(array)):
            for y in range(0, len(array)):
                if (abs(array[x, y]) > largest):
                    largest = array[x, y]
        return largest

#####################################################################
    def one_rk_step(self, nproc):
        # Subroutine to calculate one change in a runge-kutta step of any order
        # Using EOM that integrates CI coefficients, rotmat, and MF 1RDM

        # Prior to calling this routine need to update
        # MF 1RDM, rotmat and CI coefficients

        # Calculate the terms needed for time-derivative of mf-1rdm
        self.tot_system.get_frag_corr12RDM()

        self.tot_system.NOevals = np.diag(np.real(
            utils.rot1el(self.tot_system.glob1RDM, self.tot_system.NOevecs)))

        # Calculate embedding hamiltonian
        make_ham = time.time()
        self.tot_system.get_frag_Hemb()

        # Make sure Ecore for each fragment is 0 for dynamics
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        # Calculate change in propagated variables
        make_derivs = time.time()
        ddt_glob1RDM, ddt_NOevec, ddt_mf1RDM, G_site, ddt_mfRDM_check = (
            mf1rdm_timedep_mod.get_ddt_mf1rdm_serial(
                self.dG, self.tot_system, round(self.tot_system.Nele/2)))

        # Use change in mf1RDM to calculate X-matrix for each fragment
        make_xmat = time.time()
        self.tot_system.get_frag_Xmat(ddt_mf1RDM)

        change_glob1RDM = ddt_glob1RDM * self.delt
        change_NOevecs = ddt_NOevec * self.delt
        change_mf1RDM = ddt_mf1RDM * self.delt

        # Calculate change in embedding orbitals
        change_rotmat_list = []
        for frag in self.tot_system.frag_list:
            change_rotmat_list.append(
                -1j * self.delt * np.dot(frag.rotmat, frag.Xmat))

        # Calculate change in CI coefficients in parallel

        no_paralel_start = time.time()
        change_CIcoeffs_list = []

        for ifrag, frag in enumerate(self.tot_system.frag_list):
            change_CIcoeffs_list.append(applyham_wrapper(frag, self.delt))

        return change_NOevecs, change_rotmat_list, \
            change_CIcoeffs_list, change_glob1RDM, change_mf1RDM, ddt_mfRDM_check
 
 #####################################################################
    def print_data(self, current_time):
        # Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        # ####### CALCULATE OBSERVABLES OF INTEREST #######

        # Calculate DMET energy, which also includes calculation
        # of 1 & 2 RDMs and embedding hamiltonian for each fragment
        self.tot_system.get_DMET_E(self.nproc)

        # Calculate total number of electrons
        self.tot_system.get_DMET_Nele()

        # Print correlated density in the site basis
        cnt = 0
        corrdens = np.zeros(self.tot_system.Nsites)
        for frag in self.tot_system.frag_list:
            corrdens[cnt:cnt+frag.Nimp] = np.copy(
                np.diag(np.real(frag.corr1RDM[:frag.Nimp])))
            cnt += frag.Nimp
        corrdens = np.insert(corrdens, 0, current_time)
        np.savetxt(
            self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]),
            fmt_str)
        self.file_corrdens.flush()

        # Print output data
        writing_outfile = time.time()
        output = np.zeros((12+self.tot_system.Nsites))
        output[0] = current_time
        output[1] = self.tot_system.DMET_E
        output[2] = self.tot_system.DMET_Nele
        output[3] = np.real(np.trace(self.tot_system.mf1RDM))
        output[4] = np.real(np.trace(self.tot_system.frag_list[0].corr1RDM))
        output[5] = np.real(np.einsum('ppqq', self.tot_system.frag_list[0].corr2RDM))
        output[6] = np.linalg.norm(
            self.tot_system.frag_list[0].CIcoeffs)**2
        output[7] = np.linalg.norm(
            self.tot_system.frag_list[0].rotmat[:, 4])**2
        # self.tot_system.get_nat_orbs()
        if(np.allclose(self.tot_system.glob1RDM,
                       utils.adjoint(self.tot_system.glob1RDM),
                       rtol=0.0, atol=1e-14)):
            output[8] = 1
        else:
            output[8] = 0
        output[9:9+self.tot_system.Nsites] = (
            np.copy(self.tot_system.NOevals))
        output[10+self.tot_system.Nsites] = (
            np.copy(np.real(self.max_diag_global)))
        output[11+self.tot_system.Nsites] = (
            np.copy(np.imag(self.max_diag_global)))

        np.savetxt(
            self.file_output, output.reshape(1, output.shape[0]), fmt_str)
        self.file_output.flush()

        # Save total system to file for restart purposes using pickle
        file_system = open('restart_system.dat', 'wb')
        pickle.dump(self.tot_system, file_system)
        file_system.close()

#####################################################################
    def print_just_dens(self,current_time):
        
        fmt_str = '%20.8e'
        
        self.tot_system.get_DMET_E( self.nproc )
        self.tot_system.get_DMET_Nele()

        cnt = 0
        corrdens = np.zeros(self.tot_system.Nsites)
        for frag in self.tot_system.frag_list:
            corrdens[cnt:cnt+frag.Nimp] =(
                    np.diag(np.real(frag.corr1RDM[:frag.Nimp]))) 
            cnt += frag.Nimp
        corrdens = np.insert( corrdens, 0, current_time )
        np.savetxt( 
                   self.file_corrdens, 
                   corrdens.reshape(1, corrdens.shape[0]), fmt_str)
        self.file_corrdens.flush()

#####################################################################
    def import_check(self, tol):
        old_MF = np.copy(self.tot_system.mf1RDM)
        old_global = np.copy(self.tot_system.glob1RDM)
        self.tot_system.get_frag_corr1RDM()
        self.tot_system.get_glob1RDM()
        self.tot_system.get_nat_orbs()
        self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        check = print(np.allclose(old_MF, self.tot_system.mf1RDM, tol))
        check_glob = print(
            np.allclose(old_global, self.tot_system.glob1RDM, tol))
        if check is False:
            print("MF calculation doesnt agree up to the ", tol)
            print(old_MF-self.tot_system.mf1RDM)
            quit()
        if check_glob is False:
            print("Global calculation doesnt agree up to the ", tol)
            print(old_global-self.tot_system.glob1RDM)
            quit()

#####################################################################
def applyham_wrapper(frag, delt):

    # Subroutine to call pyscf to apply FCI
    # hamiltonian onto FCI vector in dynamics
    # Includes the -1j*timestep term and the
    # addition of bath-bath terms of X-matrix
    # to embedding Hamiltonian
    # The wrapper is necessary to parallelize using
    # Pool and must be separate from
    # the class because the class includes
    # IO file types (annoying and ugly but it works)

    Xmat_sml = np.zeros([2*frag.Nimp, 2*frag.Nimp], dtype=complex)
    Xmat_sml[frag.Nimp:, frag.Nimp:] = np.copy(
        frag.Xmat[frag.bathrange[:, None], frag.bathrange])
#    return -1j * delt * applyham_pyscf.apply_ham_pyscf_spinor(
#        frag.CIcoeffs, frag.h_emb-Xmat_sml, frag.V_emb,
#        frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore)

    return -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex(
        frag.CIcoeffs, frag.h_emb-Xmat_sml, frag.V_emb,
        frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore)
