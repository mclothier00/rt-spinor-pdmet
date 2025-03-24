# THIS CODE PROPAGATES FCI COEFFICIENTS USING 4TH ORDER RUNGE-KUTTA
# EVERYTHING IS IN ATOMIC UNITS
# ASSUMES CHEMISTRY NOTATION FOR THE 2 ELECTRON INTEGRALS THROUGHOUT
# THE NOTATION FOR THE 1-RDM IS AS FOLLOWS:
# gamma_ij = < a_j^dag a_i >, WHICH IS THE OPPOSITE OF PYSCF

# NOTE: if running with gen = True, must manually set Nsites as correct number
#       of spinors

import numpy as np
import real_time_pDMET.scripts.integrators as integrators
import real_time_pDMET.rtpdmet.dynamics.fci_mod as fci_mod
import sys
import real_time_pDMET.scripts.utils as utils

# NOTE: assumes an even number of electrons for the integrator


class tdfci:
    # Class to perform a time-dependent FCI calculation

    #####################################################################

    def __init__(
        self,
        Nsites,
        Nelec,
        h_site,
        V_site,
        CIcoeffs,
        delt,
        Nstep,
        Nprint,
        Ecore=0.0,
        gen=False,
    ):
        self.Nsites = Nsites
        self.Nelec = Nelec
        self.h_site = h_site
        self.V_site = V_site
        self.CIcoeffs = CIcoeffs
        self.delt = delt
        self.Nstep = Nstep
        self.Nprint = Nprint
        self.Ecore = Ecore
        self.gen = gen

        # Convert CI coefficients to complex arrays if they're not already
        if not np.iscomplexobj(self.CIcoeffs):
            self.CIcoeffs = self.CIcoeffs.astype(complex)

        # Define output files
        self.file_output = open("output.dat", "wb")
        self.file_corrdens = open("corr_density.dat", "wb")
        if self.gen:
            self.file_spindens = open("spin_density.dat", "wb")
            self.file_spinx = open("spin_x.dat", "w")
            self.file_spiny = open("spin_y.dat", "w")
            self.file_spinz = open("spin_z.dat", "w")
            self.file_totspins = open("total_spins.dat", "w")

    #####################################################################

    def kernel(self):
        # main subroutine for Real-Time propagation of FCI
        # coefficients under a time-independent hamiltonian

        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("BEGIN REAL-TIME FCI CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

        current_time = 0.0

        # DYNAMICS LOOP:
        for step in range(self.Nstep):
            # Print data before taking time-step,
            # this always prints out data at initial time step
            if np.mod(step, self.Nprint) == 0:
                print(
                    "Writing data at step ",
                    step,
                    "and time",
                    current_time,
                    "for TDFCI calculation",
                )
                self.print_data(current_time)
                sys.stdout.flush()
            if not self.gen:
                # Integrate FCI coefficients by a time-step
                self.CIcoeffs = integrators.runge_kutta_pyscf(
                    self.CIcoeffs,
                    self.Nsites,
                    int(self.Nelec / 2),
                    int(self.Nelec / 2),
                    self.delt,
                    self.h_site,
                    self.V_site,
                    self.Ecore,
                )

            if self.gen:
                # Integrate FCI coefficients by a time-step
                self.CIcoeffs = integrators.runge_kutta_pyscf(
                    self.CIcoeffs,
                    self.Nsites,
                    int(self.Nelec / 2),
                    int(self.Nelec / 2),
                    self.delt,
                    self.h_site,
                    self.V_site,
                    self.Ecore,
                    self.gen,
                )

            # update the current time
            current_time = self.delt * (step + 1)

        # Print data at final step regardless of Nprint
        print(
            "Writing data at step ",
            step + 1,
            "and time",
            current_time,
            "for TDFCI calculation",
        )
        self.print_data(current_time)
        sys.stdout.flush()

        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END REAL-TIME FCI CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

    #####################################################################

    def print_data(self, current_time):
        fmt_str = "%20.8e"

        # ####### CALCULATE OBSERVABLES OF INTEREST #######

        if not self.gen:
            # Calculate 1RDM
            corr1RDM = fci_mod.get_corr1RDM(self.CIcoeffs, self.Nsites, self.Nelec)

            file = open("restricted.txt", "w")
            file.write("from restricted: \n")
            file.write(f"correlated 1RDM: \n {corr1RDM} \n")
            file.close()

            # Calculate total energy
            Etot = fci_mod.get_FCI_E(
                self.h_site,
                self.V_site,
                self.Ecore,
                self.CIcoeffs,
                self.Nsites,
                int(self.Nelec / 2),
                int(self.Nelec / 2),
            )

            file = open("restricted.txt", "a")
            file.write(f"energy: {Etot}")
            file.close()

        if self.gen:
            # Calculate 1RDM
            corr1RDM = fci_mod.get_corr1RDM(
                self.CIcoeffs, self.Nsites, self.Nelec, self.gen
            )

            file = open("generalized.txt", "w")
            file.write("from generalized \n")
            file.write(f"from generalized: \n {corr1RDM} \n")
            file.close()

            # total spin vectors
            den = utils.reshape_gtor_matrix(corr1RDM)
            Nsp = int(self.Nsites / 2)
            ovlp = np.eye(Nsp)

            magx = np.sum((den[:Nsp, Nsp:] + den[Nsp:, :Nsp]) * ovlp)
            magy = 1j * np.sum((den[:Nsp, Nsp:] - den[Nsp:, :Nsp]) * ovlp)
            magz = np.sum((den[:Nsp, :Nsp] - den[Nsp:, Nsp:]) * ovlp)

            all_spin = np.insert(np.array([magx.real, magy.real, magz.real]), 0, current_time)
            np.savetxt(
                self.file_totspins, all_spin.reshape(1, all_spin.shape[0]), fmt_str
            )
            self.file_totspins.flush()
            
            # spin on each site

            sites_x = []
            sites_y = []
            sites_z = []

            for i in range(Nsp):
                ovlp = np.zeros((Nsp, Nsp))
                ovlp[i,i] = 1

                site_magx = np.sum((den[:Nsp, Nsp:] + den[Nsp:, :Nsp]) * ovlp)
                site_magy = 1j * np.sum((den[:Nsp, Nsp:] - den[Nsp:, :Nsp]) * ovlp)
                site_magz = np.sum((den[:Nsp, :Nsp] - den[Nsp:, Nsp:]) * ovlp)

                sites_x.append(site_magx.real)
                sites_y.append(site_magy.real)
                sites_z.append(site_magz.real)

            sites_x = np.insert(np.array(sites_x), 0, current_time)
            sites_y = np.insert(np.array(sites_y), 0, current_time)
            sites_z = np.insert(np.array(sites_z), 0, current_time)

            np.savetxt(
                self.file_spinx, sites_x.reshape(1, sites_x.shape[0]), fmt_str
            )
            np.savetxt(
                self.file_spiny, sites_y.reshape(1, sites_y.shape[0]), fmt_str
            )
            np.savetxt(
                self.file_spinz, sites_z.reshape(1, sites_z.shape[0]), fmt_str
            )
            self.file_spinx.flush()
            self.file_spiny.flush()
            self.file_spinz.flush()

            # Calculate total energy
            Etot = fci_mod.get_FCI_E(
                self.h_site,
                self.V_site,
                self.Ecore,
                self.CIcoeffs,
                self.Nsites,
                int(self.Nelec / 2),
                int(self.Nelec / 2),
                self.gen,
            )

            file = open("generalized.txt", "a")
            file.write(f"energy: {Etot}")
            file.close()

        # Calculate total number of electrons
        # (used as convergence check for time-step)
        Nele = np.real(np.sum(np.diag(corr1RDM)))

        # ####### PRINT OUT EVERYTHING #######

        diagcorr1RDM = np.real(np.diag(corr1RDM))

        # Print correlated density in the site basis
        if not self.gen:
            corrdens = diagcorr1RDM
            corrdens = np.insert(corrdens, 0, current_time)
        if self.gen:
            corrdens = diagcorr1RDM.reshape(-1, 2).sum(axis=1)
            corrdens = np.insert(corrdens, 0, current_time)

        np.savetxt(self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str)
        self.file_corrdens.flush()

        # Print spin density if generalized
        if self.gen:
            spindens = diagcorr1RDM
            spindens = np.insert(diagcorr1RDM, 0, current_time)

            np.savetxt(self.file_spindens, spindens.reshape(1, spindens.shape[0]), fmt_str)
            self.file_spindens.flush()

        # Print output data
        output = np.zeros(3)
        output[0] = current_time
        output[1] = Etot
        output[2] = Nele
        np.savetxt(self.file_output, output.reshape(1, output.shape[0]), fmt_str)
        self.file_output.flush()


#####################################################################
