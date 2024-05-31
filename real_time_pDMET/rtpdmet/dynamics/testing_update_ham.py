import pyscf
import numpy as np
import math

NL = 2
NR = 3
Ndots = 1
Nimp = 2 # frangment size
Nsites = NL+NR+Ndots
hubsite_indx = np.array([2])

A_nott = 3
t_nott = 5
omega = 4.0
curr_time = 9
U = 3.0
laser_sites = list(range(2,5))
t_d = 0.5

file_laser = open('laser.dat', 'w')

def make_genham_multi_imp_anderson_realspace_laser( N, U, laser, laser_sites=None, imp_indx=None,  t=1.0, update=False, boundary=False, Full=False ):
    ## NEED TO TEST
    # starting from alpha/beta degeneracy?

    hmat = np.zeros([2*N,2*N], dtype=np.complex_)

    # what does this do? What is "boundary"?
    if boundary is True:
        hmat[ 0, (2*N)-1 ] = -1*t*np.exp(1j*laser)

    for site in range((2*N)-1):
        if site in laser_sites:
            hmat[site, site+1] = -1*t*np.exp(1j*laser)
        else:
            hmat[site, site+1] = -1*t

    hmat = (np.triu(hmat)+np.triu(hmat,1).conjugate().transpose())
    if update is True:
        return hmat

    #Form the trivial two electron terms
    if( Full ):
        Vmat = np.zeros( [ 2*N, 2*N, 2*N, 2*N ] )
        for imp in imp_indx:
            Vmat[imp,imp,imp,imp] = U
    else:
        Vmat = U
    return hmat,Vmat


laser_pulse =(
    A_nott*np.exp(-((curr_time-t_nott)**2)/(2*t_d**2))
    *math.cos(omega*(curr_time-t_nott)))

Nimp = np.shape(hubsite_indx)[0]
h_site =(
    make_genham_multi_imp_anderson_realspace_laser(
        Nsites, U, laser_pulse, laser_sites,
        hubsite_indx, t=1, update=True,
        boundary=False, Full=False))

print(h_site)

fmt_str = '%20.8e'
laser_time = np.array([curr_time, laser_pulse])
np.savetxt( file_laser, laser_time.reshape(1,2), fmt_str)
file_laser.flush()