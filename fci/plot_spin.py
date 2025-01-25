import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

# site index starting from 1


def plot_site_mag():
    files = ["spin_x.dat", "spin_y.dat", "spin_z.dat"]

    for i, file in enumerate(files):
        mag = []

        with open(file, "r") as f:
            for line in f:
                data = line.split()
                data = [x.strip() for x in data]
                mag.append(data)

        mag = np.asarray(mag)

        sites = mag.shape[1]

        fig, ax = plt.subplots()
        for s in range(1, sites):
            plt.plot(
                mag[:, 0],
                mag[:, s],
                label=f"Mag: site {s}",
            )
            print(mag[3, s])

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("Time (au)")
        ax.set_ylabel("Magnetization (au)")
        ax.legend()
        if i == 0:
            fig.savefig("mag_x.png")
        if i == 1:
            fig.savefig("mag_y.png")
        if i == 2:
            fig.savefig("mag_z.png")


def plot_multiple_electron_site_den(
    res_filename, gen_filename, fig_filename, res_site_index, gen_site_index
):
    restable = []
    openfile = f"{res_filename}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            restable.append(data)
    restable = np.asarray(restable)

    gentable = []
    openfile = f"{gen_filename}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            gentable.append(data)
    gentable = np.asarray(gentable)

    plt.figure()
    for i in res_site_index:
        plt.plot(
            restable[:, 0].astype(complex).real,
            restable[:, i].astype(complex).real,
            label=f"Res: site {i}",
        )
    for i in gen_site_index:
        site_den = (
            gentable[:, i].astype(complex).real
            + gentable[:, (i + 1)].astype(complex).real
        )
        plt.scatter(
            gentable[:, 0].astype(complex).real,
            site_den,
            label=f"Gen: spinors {i}, {i+1}",
            s=10,
        )

    plt.xlabel("Time (au)")
    plt.ylabel("Site Density")
    plt.legend()
    plt.savefig(f"{fig_filename}")


def plot_multiple_spin_den(filename_fci, filename_dmet, fig_filename):
    table_fci = []
    openfile = f"{filename_fci}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table_fci.append(data)
    table_fci = np.asarray(table_fci)

    table_dmet = []
    openfile = f"{filename_dmet}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table_dmet.append(data)
    table_dmet = np.asarray(table_dmet)

    columns = range(table_fci.shape[1])
    if columns != range(table_dmet.shape[1]):
        print("ERROR: Files do not contain the same number of sites.")
        exit()

    alpha_site_index = columns[1::2]
    beta_site_index = columns[2::2]

    fig, ax = plt.subplots()
    for i in alpha_site_index:
        ax.plot(
            table_fci[:, 0].astype(complex).real,
            table_fci[:, i].astype(complex).real,
            label=f"spin (HF) {i}",
        )
        ax.scatter(
            table_dmet[:, 0].astype(complex).real,
            table_dmet[:, i].astype(complex).real,
            label=f"spin (DMET) {i}",
            s=10,
        )

        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        ax.set_xlabel("Time (au)")
        ax.set_ylabel("Site Density")
        ax.legend()
        fig.savefig(f"alpha_{fig_filename}")

    fig, ax = plt.subplots()
    for i in beta_site_index:
        ax.plot(
            table_fci[:, 0].astype(complex).real,
            table_fci[:, i].astype(complex).real,
            label=f"spin (HF) {i}",
        )
        ax.scatter(
            table_dmet[:, 0].astype(complex).real,
            table_dmet[:, i].astype(complex).real,
            label=f"spin (DMET) {i}",
            s=10,
        )

        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        ax.set_xlabel("Time (au)")
        ax.set_ylabel("Site Density")
        ax.legend()
        fig.savefig(f"beta_{fig_filename}")


plot_site_mag()
