### adjusted from TiDES code ###
import numpy as np


def static_bfield(hcore, bfield):
    
    x_bfield = bfield[0]
    y_bfield = bfield[1]
    z_bfield = bfield[2]

    Nsp = int(hcore.shape[0] / 2)

    hcore = hcore[:Nsp, :Nsp]

    hprime = np.zeros([2 * Nsp, 2 * Nsp], dtype=complex)
    hprime[:Nsp, :Nsp] = hcore + 0.5 * z_bfield * np.identity(Nsp)
    hprime[Nsp:, Nsp:] = hcore - 0.5 * z_bfield * np.identity(Nsp)
    hprime[Nsp:, :Nsp] = 0.5 * (x_bfield + 1j * y_bfield) * np.identity(Nsp)
    hprime[:Nsp, Nsp:] = 0.5 * (x_bfield - 1j * y_bfield) * np.identity(Nsp)
    
    return hprime
