# Multi-fragment real-time projected Density Matrix Embedding Theory 

A computational method that directly simulates non-equilibrium electron dynamics in strongly correlated systems. Examples of non-equilibrium conditions are a laser or a potential bias exerted on the system or a sudden change in the strength of Culombic interactions within the system. This method is based on the propagation of the full wavefunction of the system, therefore it allows for a calculation of any time-dependent observable. As an example, the time-dependent electron density in a 2-dimensional single impurity Anderson Model (SIAM) following a change of a Culombic interactions parameter U from U=0 to U=3 is shown below. 

This code is developed in conjunction with the theoretical work described in:

Yehorova, D., & Kretchmer, J. S. (2022). A multi-fragment real-time extension of projected density matrix embedding theory: Non-equilibrium electron dynamics in extended systems. arXiv preprint arXiv:2209.06368.
![](https://github.com/DYehorova/real_time_pDMET/blob/main/elec_density_2D.gif)

# Installation and Use 
Dependency: PySCF (https://pyscf.org/install.html)

After "real_time_pDMET" directory is added to the PYTHONPATH, example run files for the electron dynamics in 1D and 2D Anderson Impurity Models can be executed within "real_time_pDMET/rtpdmet/examples" directory.
As this method is Hamiltonian-agnostic, the application of this code is not limited to these examples and can be expanded to other model Hamiltonians. 

Additionally, this repository contains code for real-time Hartree Fock and real-time Full CI that can be used as comparison solutions. 
