import pyscf
from scipy import linalg
import numpy as np

def concat_strings(alphastr, betastr):
    matrix_elements = []
    for i in range(len(alphastr)): 
        for j in range(len(betastr)):
            # convert binary numbers to strings and remove prefix
            alpha_str = bin(alphastr[i])[2:]
            beta_str = bin(betastr[j])[2:]
        
            # adds leading zeros so both strings are of the same length
            alpha_str = alpha_str.zfill(3)
            beta_str = beta_str.zfill(3)
        
            # concatenate strings
            matrix_str = "".join(i for j in zip(beta_str, alpha_str) for i in j)
            matrix_elements.append(int('0b' + matrix_str, 2))
    return matrix_elements

def to_gen_coeff(norb_alpha, norb_beta, norb_gen, nalpha, nbeta, coeffs):
    nelec = nalpha + nbeta

    # spinor indices
    strings = pyscf.fci.cistring.make_strings(np.arange(norb_gen), nelec)
    print(f'coefficient strings: {[bin(x) for x in strings]}')

    # matrix row, column indices
    alphastr = pyscf.fci.cistring.make_strings(np.arange(norb_alpha), nalpha)
    betastr = pyscf.fci.cistring.make_strings(np.arange(norb_beta), nbeta)
    print(f'{[bin(x) for x in alphastr]} \n {[bin(z) for z in betastr]}')

    # matrix elements 

    matrix_elements = concat_strings(alphastr, betastr)
    print(f'new coefficient strings from matrix: {[bin(x) for x in matrix_elements]}')


    ### mapping from newly created strings to strings from make_strings
    new_coeff = np.zeros(len(strings), dtype=complex)
    coeffs = np.matrix.flatten(coeffs)

    for i in range(len(coeffs)):
        index = pyscf.fci.cistring.str2addr(norb_gen, nelec, matrix_elements[i])
        new_coeff[index] = coeffs[i]

    return new_coeff

mol = pyscf.M(
            atom = 'H 0 0 0; H 0 0 1.1; H 1.1 0 0',  # in Angstrom
            basis = 'sto3g',
            symmetry = True,
            spin = 1
            )
myhf = mol.RHF().run()


### FCI calculations
# UHF
myuhf = mol.UHF().run()
cisolver = pyscf.fci.FCI(myuhf)
print('E(UHF-FCI) = %.12f' % cisolver.kernel()[0])
print('Coeff(UHF):')
UHF_coeffs = cisolver.kernel()[1]
print(UHF_coeffs)

# GHF
myghf = mol.GHF().run()
cisolver = pyscf.fci.FCI(myghf)
print('E(GHF-FCI) = %.12f' % cisolver.kernel()[0])
print('Coeff(GHF):')
print(cisolver.kernel()[1])


### string indexing
norb_alpha = norb_beta = 3
norb_gen = 6
nelec = 3
nalpha = 2
nbeta = 1

new_coeff = to_gen_coeff(norb_alpha, norb_beta, norb_gen, nalpha, nbeta, UHF_coeffs)

print(new_coeff)
