import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


def plot_fci_dmet_site_mag(dmet_filename, fci_filename, figname):
    fci = []
    dmet = []

    with open(dmet_filename, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            dmet.append(data)

    with open(fci_filename, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            fci.append(data)

    dmet = np.asarray(dmet, dtype=float)
    fci = np.asarray(fci, dtype=float)

    fig, ax = plt.subplots()

<<<<<<< HEAD
    sites = range(dmet.shape[1])
=======
    sites = range(1, dmet.shape[1]-1)
>>>>>>> fcb5cbc9a46b4256411eb62ced5175e0861d5a19
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sites)))
    fig, ax = plt.subplots()
    
    for i, color in zip(sites, colors):
        ax.scatter(
            dmet[:, 0][::5],
            dmet[:, i][::5],
            label=f"DMET mag: site {i}",
            marker="o",
            color=color,
            facecolors='none',
            edgecolors=color
        )
        ax.plot(
            fci[:, 0],
            fci[:, i],
            label=f"FCI mag: site {i}",
            color=color,
    
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Time (au)")
    ax.set_ylabel("Magnetization (au)")
    ax.legend()
    
    fig.savefig(f"{figname}")
    

def plot_mag():
    magx = []
    magy = []
    magz = []
    
    with open("spin_x.dat", "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            magx.append(data)

    with open("spin_y.dat", "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            magy.append(data)

    with open("spin_z.dat", "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            magz.append(data)

    magx = np.asarray(magx, dtype=float)
    magy = np.asarray(magy, dtype=float)
    magz = np.asarray(magz, dtype=float)
    
    fig, ax = plt.subplots()
   
    ax.plot(
        magx[:, 0],
        magx[:, 1],
        label=f"Mag x",
        color='firebrick',
    )
    ax.plot(
        magy[:, 0],
        magy[:, 1],
        label=f"Mag y",
        color='cornflowerblue',
    )
    ax.plot(
        magz[:, 0],
        magz[:, 1],
        label=f"Mag z",
        color='forestgreen',
    )

    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Time (au)")
    ax.set_ylabel("Magnetization (au)")
    ax.legend()

    fig.savefig("magnetization.png")


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


def plot_hf_dmet_spin_den(filename_hf, filename_dmet, fig_filename):
    table_hf = []
    openfile = f"{filename_hf}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table_hf.append(data)
    table_hf = np.asarray(table_hf)

    table_dmet = []
    openfile = f"{filename_dmet}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table_dmet.append(data)
    table_dmet = np.asarray(table_dmet)

    columns = range(table_hf.shape[1])
    if columns != range(table_dmet.shape[1]):
        print("ERROR: Files do not contain the same number of sites.")
        exit()

    alpha_dmet = columns[1::2]
    beta_dmet = columns[2::2]

    fig, ax = plt.subplots()
    for i in range(1, (1 + table_hf.shape[1] // 2)):
        ax.plot(
            table_hf[:, 0].astype(complex).real,
            table_hf[:, i].astype(complex).real,
            label=f"spin (HF) {i}",
        )

    for i in alpha_dmet:
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
    for i in range((1 + table_hf.shape[1] // 2), table_hf.shape[1]):
        ax.plot(
            table_hf[:, 0].astype(complex).real,
            table_hf[:, i].astype(complex).real,
            label=f"spin (HF) {i}",
        )

    for i in beta_dmet:
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


def plot_fci_dmet_spin_den(filename_fci, filename_dmet, fig_filename):
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
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(beta_site_index)))

    fig, ax = plt.subplots(figsize=(8,6))
    
    for i, color in zip(alpha_site_index, colors):
        ax.plot(
            table_fci[:, 0].astype(complex).real,
            table_fci[:, i].astype(complex).real,
            color=color,
        )
        ax.scatter(
            table_dmet[:, 0].astype(complex).real[::5],
            table_dmet[:, i].astype(complex).real[::5],
            marker="o",
            color=color,
            facecolors='none',
            edgecolors=color
        )

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.tick_params(axis="both", labelsize=16, width=2)

        solid_line_handle = Line2D([0], [0], color="navy", lw=2, label="FCI")
        bubbles_handle = Line2D(
            [0], [0], color="navy", marker='o', markerfacecolor='none', markeredgecolor='navy', label="RT-pDMET"
        )

        ax.legend(handles=[solid_line_handle, bubbles_handle], fontsize=17, edgecolor="black")

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_xlabel("Time (au)", fontsize=17)
        ax.set_ylabel("'Alpha' Site Density", fontsize=17)
        fig.savefig(f"alpha_{fig_filename}", dpi=300)

    fig, ax = plt.subplots(figsize=(8,6))
    for i, color in zip(beta_site_index, colors):
        ax.plot(
            table_fci[:, 0].astype(complex).real,
            table_fci[:, i].astype(complex).real,
            color=color,
        )
        ax.scatter(
            table_dmet[:, 0].astype(complex).real[::5],
            table_dmet[:, i].astype(complex).real[::5],
            marker="o",
            color=color,
            facecolors='none',
            edgecolors=color
        )

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.tick_params(axis="both", labelsize=16, width=2)

        solid_line_handle = Line2D([0], [0], color="navy", lw=2, label="FCI")
        bubbles_handle = Line2D(
            [0], [0], color="navy", marker='o', markerfacecolor='none', markeredgecolor='navy', label="RT-pDMET"
        )

        ax.legend(handles=[solid_line_handle, bubbles_handle], fontsize=17, edgecolor="black")
        

        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        ax.set_xlabel("Time (au)", fontsize=17)
        ax.set_ylabel("'Beta' Site Density", fontsize=17)
        fig.savefig(f"beta_{fig_filename}", dpi=300)


def plot_spin_diff(filename_fci, filename_dmet, fig_filename):
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
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(beta_site_index)))

    fig, ax = plt.subplots(figsize=(8,6))
    
    for i, color in zip(alpha_site_index, colors):
        ax.plot(
            table_fci[:, 0].astype(complex).real,
            (table_fci[:, i].astype(complex).real - table_dmet[:, i].astype(complex).real),
            color=color,
        )

        #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.tick_params(axis="both", labelsize=16, width=2)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_xlabel("Time (au)", fontsize=17)
        ax.set_ylabel("'Alpha' Site Density Error", fontsize=17)
        fig.savefig(f"alpha_{fig_filename}_error", dpi=300)

    fig, ax = plt.subplots(figsize=(8,6))
    for i, color in zip(beta_site_index, colors):
        ax.plot(
            table_fci[:, 0].astype(complex).real,
            (table_fci[:, i].astype(complex).real - table_dmet[:, i].astype(complex).real),
            color=color,
        )

        ax.tick_params(axis="both", labelsize=16, width=2)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_xlabel("Time (au)", fontsize=17)
        ax.set_ylabel("'Beta' Site Density Error", fontsize=17)
        fig.savefig(f"beta_{fig_filename}_error.png", dpi=300)

<<<<<<< HEAD
dmet_filename = "dmetx.dat"
fci_filename = "fcix.dat"
plotname = "magx.png"

plot_fci_dmet_site_mag(dmet_filename, fci_filename, plotname)
=======
>>>>>>> fcb5cbc9a46b4256411eb62ced5175e0861d5a19

plot_fci_dmet_site_mag("dmetx.dat", "fcix.dat", "magx.png")
plot_fci_dmet_site_mag("dmety.dat", "fciy.dat", "magy.png")
plot_fci_dmet_site_mag("dmetz.dat", "fciz.dat", "magz.png")

plot_fci_dmet_spin_den("spin_density.dat", "electron_density.dat", "spinor")
