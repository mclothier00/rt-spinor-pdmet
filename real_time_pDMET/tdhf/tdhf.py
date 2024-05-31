import numpy as np
import sys
import real_time_pDMET.scripts.integrators as integrators

#####################################################################


class tdhf():
    # Class to perform a time-dependent hartree-fock calculation
    #####################################################################

    def __init__(self, Nsites, Nelec, h_site, mf1RDM, delt, Nstep, Nprint):

        self.Nsites = Nsites
        self.Nelec = Nelec
        self.h_site = h_site
        self.mf1RDM = mf1RDM
        self.delt = delt
        self.Nstep = Nstep
        self.Nprint = Nprint

        # Convert rotation matrices and CI
        # coefficients to complex arrays if they're not already
        if(not np.iscomplexobj(self.mf1RDM)):
            self.mf1RDM = self.mf1RDM.astype(complex)

        # Define output files
        self.file_output = open('output.dat', 'w')
        self.file_corrdens = open('corr_density.dat', 'w')
    #####################################################################

    def kernel(self):

        current_time = 0.0

        for step in range(self.Nstep):

            # Print data before taking time-step,
            # this always prints out data at initial time step
            if(np.mod(step, self.Nprint) == 0):
                print('Writing data at step ', step,
                      'and time', current_time, 'for TDHF calculation')
                self.print_data(current_time)
                sys.stdout.flush()

            # Integrate the mean-field 1RDM
            # exactly assuming time-indep hamiltonian
            self.mf1RDM = integrators.exact_timeindep_1rdm(
                self.mf1RDM, self.h_site, self.delt)

            # Increase current_time
            current_time += self.delt

        # Print data at final step regardless of Nprint
        print('Writing data at step ', step+1,
              'and time', current_time, 'for RT-pDMET calculation')
        self.print_data(current_time)
        sys.stdout.flush()

        # Close output files
        self.file_output.close()
        self.file_corrdens.close()
    #####################################################################

    def print_data(self, current_time):
        # Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        # ####### CALCULATE OBSERVABLES OF INTEREST #######

        # Calculate total energy
        Etot = np.real(np.einsum('ij,ji', self.h_site, self.mf1RDM))

        # Calculate total number of electrons
        Nele = np.real(np.sum(np.diag(self.mf1RDM)))

        # ####### PRINT OUT EVERYTHING #######

        # Print correlated density in the site basis
        corrdens = np.real(np.diag(self.mf1RDM))
        corrdens = np.insert(corrdens, 0, current_time)
        np.savetxt(self.file_corrdens,
                   corrdens.reshape(1, corrdens.shape[0]), fmt_str)
        self.file_corrdens.flush()

        # Print output data
        output = np.zeros(3)
        output[0] = current_time
        output[1] = Etot
        output[2] = Nele
        np.savetxt(self.file_output,
                   output.reshape(1, output.shape[0]), fmt_str)
        self.file_output.flush()
    #####################################################################
