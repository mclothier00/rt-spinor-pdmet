import numpy as np
from scipy.linalg import eigh, inv
from pyscf import gto, scf
import scipy
import matplotlib.pyplot as plt

# class needs mf, timestep, frequency, total_steps


class GHF:
    def __init__(
        self, mf, timestep, frequency, total_steps, filename, usite, sites, orth=None
    ):
        self.timestep = timestep
        self.frequency = frequency
        self.total_steps = total_steps
        self.filename = filename
        self._scf = mf
        self.usite = usite
        self.sites = sites

        if orth is None:
            self.orth = scf.addons.canonical_orth_(self._scf.get_ovlp())

    def rdm1_ghf_to_rhf(rdm1, sites):
        rdm1_aa = rdm1[:sites, :sites]
        rdm1_bb = rdm1[sites:, sites:]
        return rdm1_aa + rdm1_bb

    ####### DYNAMICS #######
    def dynamics(self):
        ### creating output file for observables (edit to be main output file and to adjust what is calculated)
        observables = open(f"{self.filename}.txt", "w")
        #        with open(F'{self.filename}.txt', 'a') as f:
        #            observables.write('{0: >20}'.format('Time'))
        #            observables.write('{0: >35}'.format('Mag x'))
        #            observables.write('{0: >36}'.format('Mag y'))
        #            observables.write('{0: >37}'.format('Mag z'))
        #            observables.write('{0: >37}'.format('Energy'))
        #            observables.write(F'\n')
        #            observables.flush()

        ### creating initial core hamiltonian
        fock = self._scf.get_fock()
        mag_x = []
        mag_y = []
        mag_z = []
        t_array = []
        energy = []

        fmt_str = "%20.8e"

        ovlp = self._scf.get_ovlp()
        hcore = self._scf.get_hcore()
        Nsp = int(ovlp.shape[0] / 2)
        veff = self._scf.get_veff()

        for i in range(0, self.total_steps):
            ### transforming coefficients into an orthogonal matrix
            mo_oth = np.dot(inv(self.orth), self._scf.mo_coeff)

            ### create transformation matrix U from Fock matrix at time t
            fock_oth = np.dot(self.orth.T, np.dot(fock, self.orth))

            u = scipy.linalg.expm(-1j * 2 * self.timestep * fock_oth)

            ### propagate MO coefficients
            if i != 0:
                mo_oth_new = np.dot(u, mo_oth_old)
            else:
                mo_oth_new = np.dot(u, mo_oth)

            ### transform coefficients back into non-orthogonal basis and get density matrix
            self._scf.mo_coeff = np.dot(self.orth, mo_oth_new)
            den = self._scf.make_rdm1()

            # calculate a new fock matrix
            fock = self._scf.get_fock(hcore, veff)

            # calculate energy and other observables
            if np.mod(i, self.frequency) == 0:
                ener_tot = self._scf.energy_tot()

                den = self._scf.make_rdm1()
                t = i * self.timestep

                den_diag = np.real(np.diag(den))
                site_den = np.zeros(Nsp)
                for i in range(0, Nsp):
                    site_den[i] = den_diag[i] + den_diag[i + Nsp]
                site_den = np.insert(site_den, 0, t)

                with open(f"{self.filename}.txt", "a") as f:
                    np.savetxt(f, site_den.reshape(1, site_den.shape[0]), fmt_str)

            # checking final denisty matrix:
            if i == (self.total_steps - 1):
                den = self._scf.make_rdm1()

                rdm1_aa = den[: self.sites, : self.sites]
                rdm1_bb = den[self.sites :, self.sites :]
                new_den = rdm1_aa + rdm1_bb

                print(f"final density matrix: \n {new_den}")

            #                mag_x_value = 0
            #                mag_y_value = 0
            #                mag_z_value = 0
            #
            #                for k in range(0, Nsp):
            #                    for j in range(0, Nsp):
            #                        ab_add = den[:Nsp, Nsp:][k,j] + den[Nsp:, :Nsp][k,j]
            #                        mag_x_value += ab_add * ovlp[k,j]
            #
            #                        ab_sub = den[:Nsp, Nsp:][k,j] - den[Nsp:, :Nsp][k,j]
            #                        mag_y_value += ab_sub * ovlp[k,j]
            #
            #                        aa_bb = den[:Nsp, :Nsp][k,j] - den[Nsp:, Nsp:][k,j]
            #                        mag_z_value += aa_bb * ovlp[k,j]
            #
            #                mag_y_value = 1j * mag_y_value
            #
            #                t = (i * self.timestep) / 41341.374575751
            #
            #                with open(F'{self.filename}.txt', 'a') as f:
            #                    observables.write(F'{t:20.8e} \t {mag_x_value:20.8e} \t {mag_y_value:20.8e} \t {mag_z_value:20.8e} \t {ener_tot:20.8e} \n')
            #                    observables.flush()
            #
            mo_oth_old = mo_oth

    ####### PLOTTING RESULTS #######
    def plot_mag(self):
        table = []
        openfile = f"{self.filename}.txt"

        with open(openfile, "r") as f:
            next(f)
            for line in f:
                data = line.split("\t")
                data = [x.strip() for x in data]
                table.append(data)

        table = np.asarray(table)

        plt.figure(1)
        plt.plot(table[:, 0], np.real(table[:, 1]), "r", label="mag_x")
        plt.plot(table[:, 0], np.real(table[:, 2]), "b", label="mag_y")
        plt.plot(table[:, 0], np.real(table[:, 3]), "g", label="mag_z")
        plt.xlabel("Time (ps)")
        plt.ylabel("Magnetization (au)")
        plt.legend()
        plt.savefig(f"{self.filename}_mag.png")

    def plot_energy(self):
        table = []
        openfile = f"{self.filename}.txt"

        with open(openfile, "r") as f:
            next(f)
            for line in f:
                data = line.split("\t")
                data = [x.strip() for x in data]
                table.append(data)

        table = np.asarray(table)

        plt.figure(2)
        plt.plot(table[:, 0], np.real(table[:, 4]), "r")
        plt.xlabel("Time (ps)")
        plt.ylabel("Energy (Hartrees)")
        plt.savefig(f"{self.filename}_energy.png")

    def plot_site_den(self):
        table = []
        openfile = f"{self.filename}.txt"

        with open(openfile, "r") as f:
            for line in f:
                data = line.split()
                data = [x.strip() for x in data]
                table.append(data)

        table = np.asarray(table)

        plt.figure()
        print(f"size of usite: {self.usite}")
        exit()
        for i in self.usite:
            plt.plot(table[:, 0].astype(complex).real, table[:, i].astype(complex).real)
        plt.xlabel("Time (au)")
        plt.ylabel("Site Density")
        plt.savefig(f"site_density_ghf.png")
