#!/usr/bin/python

import numpy as np
import scipy.linalg as la
import scipy.special
import itertools

#####################################################################


def diagonalize(H, S=None):
    # subroutine to solve the general eigenvalue problem HC=SCE
    # returns the matrix of eigenvectors C, and a 1-d array of eigenvalues
    # NOTE that H must be Hermitian

    if S is None:
        S = np.identity(len(H))

    E, C = la.eigh(H, S)

    return E, C


#####################################################################


def rot1el(h_orig, rotmat):
    # subroutine to rotate one electron integrals

    tmp = np.dot(h_orig, rotmat)
    if np.iscomplexobj(rotmat):
        h_rot = np.dot(rotmat.conjugate().transpose(), tmp)
    else:
        h_rot = np.dot(rotmat.transpose(), tmp)

    return h_rot


#####################################################################


def rot2el_chem(V_orig, rotmat):
    # subroutine to rotate two electron integrals,
    # V_orig must be in chemist notation
    # V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns

    if np.iscomplexobj(rotmat):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()

    V_new = np.einsum("trus,sy -> truy", V_orig, rotmat)
    # V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum("vu,truy -> trvy", rotmat_conj, V_new)
    # V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum("trvy,rx -> txvy", V_new, rotmat)
    # V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum("wt,txvy -> wxvy", rotmat_conj, V_new)
    # V_new now Ns x Ns x Ns x Ns

    return V_new


#####################################################################

# NOTE: this should work for spinor basis; see above


def rot2el_phys(V_orig, rotmat):
    # subroutine to rotate two electron integrals,
    # V_orig must be in physics notation
    # Returns V_new in physics notation

    # V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns

    if np.iscomplexobj(rotmat):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()

    V_new = np.einsum("turs,sy -> tury", V_orig, rotmat)
    # V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum("tury,rx -> tuxy", V_new, rotmat)
    # V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum("vu,tuxy -> tvxy", rotmat_conj, V_new)
    # V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum("wt,tvxy -> wvxy", rotmat_conj, V_new)
    # V_new now Ns x Ns x Ns x Ns

    return V_new


#####################################################################


def commutator(Mat1, Mat2):
    # subroutine to calculate the commutator of two matrices
    return np.dot(Mat1, Mat2) - np.dot(Mat2, Mat1)


#####################################################################


def matprod(Mat1, *args):
    # subroutine to calculate matrix product of arbitrary number of matrices
    Result = Mat1
    for Mat in args:
        Result = np.dot(Result, Mat)
    return Result


#####################################################################


def adjoint(Mat):
    # subroutine to calculate the conjugate transpose (ie adjoint) of a matrix
    return np.conjugate(np.transpose(Mat))


#####################################################################


def chemps2_to_pyscf_CIcoeffs(CIcoeffs_chemps2, Norbs, Nalpha, Nbeta):
    # subroutine to unpack the 1d vector of
    # CI coefficients obtained from a FCI calculation using CheMPS2
    # to the correctly formatted 2d-array of CI coefficients for use with pyscf

    Nalpha_string = scipy.special.binom(Norbs, Nalpha)
    Nbeta_string = scipy.special.binom(Norbs, Nbeta)
    CIcoeffs_pyscf = np.reshape(
        CIcoeffs_chemps2, (Nalpha_string, Nbeta_string), order="F"
    )
    return CIcoeffs_pyscf


#####################################################################


def matrix2array(mat, diag=False):
    # Subroutine to flatten a symmetric matrix into a 1d array
    # Returns a 1d array corresponding
    # to the upper triangle of the symmetric matrix
    # if diag=True, all diagonal elements of the matrix should be the same
    # and first index of 1d array will be the diagonal term,
    # and the rest the upper triagonal of the matrix

    if diag:
        array = mat[np.triu_indices(len(mat), 1)]
        array = np.insert(array, 0, mat[0, 0])
    else:
        array = mat[np.triu_indices(len(mat))]

    return array


#####################################################################


def array2matrix(array, diag=False):
    # Subroutine to unpack a 1d array into a symmetric matrix
    # Returns a symmetric matrix
    # if diag=True, all diagonal elements of the returned matrix will
    # be the same corresponding to the first element of the 1d array

    if diag:
        dim = (1.0 + np.sqrt(1 - 8 * (1 - len(array)))) / 2.0
    else:
        dim = (-1.0 + np.sqrt(1 + 8 * len(array))) / 2.0

    mat = np.zeros([dim, dim])

    if diag:
        mat[np.triu_indices(dim, 1)] = array[1:]
        np.fill_diagonal(mat, array[0])
    else:
        mat[np.triu_indices(dim)] = array

    mat = mat + mat.transpose() - np.diag(np.diag(mat))

    return mat


#####################################################################


def matrix2array_nosym(mat, diag=False):
    # Subroutine to flatten a general matrix into a 1d array
    # Returns a 1d array corresponding
    # to the upper triangle of the symmetric matrix
    # if diag=True, all diagonal elements of the matrix should be the same
    # and first index of 1d array will be the diagonal term,
    # and the rest the upper triagonal of the matrix

    if diag:
        array = mat[np.triu_indices(len(mat), 1)]
        array = np.insert(array, 0, mat[0, 0])
    else:
        array = mat[np.triu_indices(len(mat))]
    return array


#####################################################################


def printarray(array, filename="array.dat", long_fmt=False):
    # subroutine to print out an ndarry
    # of 2,3 or 4 dimensions to be read by humans

    # NOTE: switched 'w' to 'a' for hard coding checks
    dim = len(array.shape)
    filehandle = open(filename, "a")
    comp_log = np.iscomplexobj(array)

    if comp_log:
        if long_fmt:
            # fmt_str = '%20.8e%+.8ej'
            fmt_str = "%25.14e%+.14ej"
        else:
            fmt_str = "%10.4f%+.4fj"
    else:
        if long_fmt:
            fmt_str = "%20.8e"
        else:
            fmt_str = "%8.4f"
    if dim == 1:
        Ncol = 1
        np.savetxt(filehandle, array, fmt_str * Ncol)

    elif dim == 2:
        Ncol = array.shape[1]
        np.savetxt(filehandle, array, fmt_str * Ncol)

    elif dim == 3:
        for dataslice in array:
            Ncol = dataslice.shape[1]
            np.savetxt(filehandle, dataslice, fmt_str * Ncol)
            filehandle.write("\n")

    elif dim == 4:
        for i in range(array.shape[0]):
            for dataslice in array[i, :, :, :]:
                Ncol = dataslice.shape[1]
                np.savetxt(filehandle, dataslice, fmt_str * Ncol)
                filehandle.write("\n")
            filehandle.write("\n")
    else:
        print("ERROR: Input array for printing is not of dimension 2, 3, or 4")
        exit()
    filehandle.close()


#####################################################################


def readarray(filename="array.dat"):
    # subroutine to read in arrays generated
    # by the printarray subroutine defined above
    # currently only works with 1d or 2d arrays

    array = np.loadtxt(filename, dtype=np.complex128)
    chk_cmplx = np.any(np.iscomplex(array))

    if not chk_cmplx:
        array = np.copy(np.real(array))
    return array


#####################################################################


def reshape_rtog_matrix(a):
    ## reshape a block diagonal matrix a to a generalized form with 1a,1b,2a,2b, etc.

    num_rows, num_cols = a.shape
    block_indices = np.arange(num_cols)
    spin_block_size = int(num_cols / 2)

    alpha_block = block_indices[:spin_block_size]
    beta_block = block_indices[spin_block_size:]

    indices = [list(itertools.chain(i)) for i in zip(alpha_block, beta_block)]

    indices = np.asarray(indices).reshape(-1)

    new_a = a[:, indices]
    new_a = new_a[indices, :]

    return new_a


#####################################################################


def reshape_gtor_matrix(a):
    ## reshape a generalized matrix with 1a,1b,2a,2b, etc. indices to blocks

    num_rows, num_cols = a.shape
    spin_block_size = num_cols // 2

    # original block indices for both rows and columns
    rows = np.arange(num_rows)
    indices_even = rows[::2]  
    indices_odd = rows[1::2]
    new_indices = np.concatenate((indices_even, indices_odd))

    # Create new, spin-blocked matrix
    new_a = a[new_indices]
    new_a = new_a[:, new_indices]

    return new_a


#####################################################################


def reshape_rtog_tensor(a):
    ## reshape a block diagonal tensor a to a generalized form with columns as 1a,1b,2a,2b, etc.

    num_rows, num_cols, dim1, dim2 = a.shape
    block_indices = np.arange(num_cols)
    spin_block_size = int(num_cols / 2)

    alpha_block = block_indices[:spin_block_size]
    beta_block = block_indices[spin_block_size:]

    indices = [list(itertools.chain(i)) for i in zip(alpha_block, beta_block)]

    indices = np.asarray(indices).reshape(-1)

    new_a = a[:, :, :, indices]
    new_a = new_a[:, :, indices, :]
    new_a = new_a[:, indices, :, :]
    new_a = new_a[indices, :, :, :]

    return new_a


#####################################################################


## NOTE: currently not sure if this correctly handles symmetries!! something we need
## to consider
def block_tensor(a):
    ## creates a "block diagonal" tensor from a given tensor

    shape1, shape2, shape3, shape4 = np.shape(a)
    ten_block = np.zeros((shape1 * 2, shape2 * 2, shape3 * 2, shape4 * 2))
    ten_block[:shape1, :shape2, :shape3, :shape4] = a[:, :, :, :]
    ten_block[shape1:, shape2:, shape3:, shape4:] = a[:, :, :, :]

    return ten_block


#####################################################################


def reshape_cicoeffs_toblock(cicoeffs):
    """
    reshapes a set of [1a, 1b, 2a, 2b, ...] CI coefficients into
    [1a, 2a, ..., 1b, 2b, ...] CI coefficients
    """

    return cicoeffs
