import numpy as np


def subtract_matrices(file0, file1, value=1e-09):
    with np.load(file0) as data0, np.load(file1) as data1:
        matrices0 = {key: value for key, value in data0.items()}
        matrices1 = {key: value for key, value in data1.items()}

    for key in matrices0.keys():
        if matrices0.keys() != matrices1.keys():
            raise ValueError(
                "Both .npz files must contain the same set of matrix names."
            )
        else:
            matrix0 = matrices0[key]
            matrix1 = matrices1[key]

            result = matrix0 - matrix1

            result[result < value] = 0

            print(f"Processed Matrix '{key}':")
            print(result)
            print()


subtract_matrices("4site_hardcode.npz", "4site_dmet.npz")
