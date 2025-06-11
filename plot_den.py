import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# site index starting from 1


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

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(gen_site_index)))

    fig, ax = plt.subplots(figsize=(8,6))
    for i, color in zip(res_site_index, colors):
        ax.plot(
            restable[:, 0].astype(complex).real,
            restable[:, i].astype(complex).real,
            color=color,
        )
    for i, color in zip(gen_site_index, colors):
        site_den = (
            gentable[:, i].astype(complex).real
            + gentable[:, (i + 1)].astype(complex).real
        )
        ax.scatter(
            gentable[:, 0].astype(complex).real[::5],
            site_den[::5],
            marker="o",
            color=color,
            facecolors='none',
            edgecolors=color
        )
        
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="both", labelsize=16, width=2)

    solid_line_handle = Line2D([0], [0], color="navy", lw=2, label="restricted")
    bubbles_handle = Line2D(
            [0], [0], color="navy", marker='o', markerfacecolor='none', markeredgecolor='navy', label="generalized"
        )

    ax.legend(handles=[solid_line_handle, bubbles_handle], fontsize=17, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel("Time (au)", fontsize=17)
    ax.set_ylabel("'Alpha' Site Density", fontsize=17)
    fig.savefig(f"{fig_filename}", dpi=300)


def plot_two_gen_electron_site_den(
    gen1, gen2, fig_filename, gen_site_index
):
    gentable1 = []
    openfile = f"{gen1}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            gentable1.append(data)
    gentable1 = np.asarray(gentable1)

    gentable2 = []
    openfile = f"{gen2}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            gentable2.append(data)
    gentable2 = np.asarray(gentable2)

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(gen_site_index)))

    fig, ax = plt.subplots(figsize=(8,6))
    for i, color in zip(gen_site_index, colors):
        site_den1 = (
            gentable1[:, i].astype(complex).real
            + gentable1[:, (i + 1)].astype(complex).real
        )
        ax.plot(
            gentable1[:, 0].astype(complex).real[::5],
            site_den1[::5],
            color=color
        )
        site_den2 = (
            gentable2[:, i].astype(complex).real
            + gentable2[:, (i + 1)].astype(complex).real
        )
        ax.scatter(
            gentable2[:, 0].astype(complex).real[::5],
            site_den2[::5],
            marker="D",
            color=color,
            facecolors='none',
            edgecolors=color
        )

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="both", labelsize=16, width=2)

    solid_line_handle = Line2D([0], [0], color="navy", lw=2, label="generalized1")
    diamond_handle = Line2D(
            [0], [0], color="navy", marker='D', markerfacecolor='none', markeredgecolor='navy', label="generalized2"
        )

    ax.legend(handles=[solid_line_handle, diamond_handle], fontsize=17, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel("Time (au)", fontsize=17)
    ax.set_ylabel("Electron Site Density", fontsize=17)
    fig.savefig(f"{fig_filename}", dpi=300)


def plot_two_res_electron_site_den(
    res1, res2, fig_filename, res_site_index
):
    restable1 = []
    openfile = f"{res1}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            restable1.append(data)
    restable1 = np.asarray(restable1)

    restable2 = []
    openfile = f"{res2}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            restable2.append(data)
    restable2 = np.asarray(restable2)

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(res_site_index)))

    fig, ax = plt.subplots(figsize=(8,6))
    for i, color in zip(res_site_index, colors):
        ax.plot(
            restable1[:, 0].astype(complex).real,
            restable1[:, i].astype(complex).real,
            color=color,
        )
        ax.scatter(
            restable2[:, 0].astype(complex).real[::5],
            restable2[:, i].astype(complex).real[::5],
            marker="v",
            color=color,
            facecolors='none',
            edgecolors=color
        )


    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="both", labelsize=16, width=2)

    solid_line_handle = Line2D([0], [0], color="navy", lw=2, label="restricted1")
    triangle_handle = Line2D(
            [0], [0], color="navy", marker='v', markerfacecolor='none', markeredgecolor='navy', label="restricted2"
        )

    ax.legend(handles=[solid_line_handle, triangle_handle], fontsize=17, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel("Time (au)", fontsize=17)
    ax.set_ylabel("Electron Site Density", fontsize=17)
    fig.savefig(f"{fig_filename}", dpi=300)



def plot_four_electron_site_den(
    res1, gen1, res2, gen2, fig_filename, res_site_index, gen_site_index
):
    restable1 = []
    openfile = f"{res1}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            restable1.append(data)
    restable1 = np.asarray(restable1)

    gentable1 = []
    openfile = f"{gen1}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            gentable1.append(data)
    gentable1 = np.asarray(gentable1)

    restable2 = []
    openfile = f"{res2}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            restable2.append(data)
    restable2 = np.asarray(restable2)

    gentable2 = []
    openfile = f"{gen2}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            gentable2.append(data)
    gentable2 = np.asarray(gentable2)

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(gen_site_index)))

    fig, ax = plt.subplots(figsize=(8,6))
    for i, color in zip(res_site_index, colors):
        ax.plot(
            restable1[:, 0].astype(complex).real,
            restable1[:, i].astype(complex).real,
            color=color,
        )
        ax.scatter(
            restable2[:, 0].astype(complex).real[::5],
            restable2[:, i].astype(complex).real[::5],
            marker="v",
            color=color,
            facecolors='none',
            edgecolors=color
        )

    for i, color in zip(gen_site_index, colors):
        site_den1 = (
            gentable1[:, i].astype(complex).real
            + gentable1[:, (i + 1)].astype(complex).real
        )
        ax.scatter(
            gentable1[:, 0].astype(complex).real[::5],
            site_den1[::5],
            marker="o",
            color=color,
            facecolors='none',
            edgecolors=color
        )
        site_den2 = (
            gentable2[:, i].astype(complex).real
            + gentable2[:, (i + 1)].astype(complex).real
        )
        ax.scatter(
            gentable2[:, 0].astype(complex).real[::5],
            site_den2[::5],
            marker="D",
            color=color,
            facecolors='none',
            edgecolors=color
        )

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="both", labelsize=16, width=2)

    solid_line_handle = Line2D([0], [0], color="navy", lw=2, label="restricted1")
    bubbles_handle = Line2D(
            [0], [0], color="navy", marker='o', markerfacecolor='none', markeredgecolor='navy', label="generalized1"
        )
    triangle_handle = Line2D(
            [0], [0], color="navy", marker='v', markerfacecolor='none', markeredgecolor='navy', label="restricted2"
        )
    diamond_handle = Line2D(
            [0], [0], color="navy", marker='D', markerfacecolor='none', markeredgecolor='navy', label="generalized2"
        )
    

    ax.legend(handles=[solid_line_handle, bubbles_handle, triangle_handle, diamond_handle], fontsize=17, edgecolor="black")

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel("Time (au)", fontsize=17)
    ax.set_ylabel("'Alpha' Site Density", fontsize=17)
    fig.savefig(f"{fig_filename}", dpi=300)


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
            label=f"spin (FCI) {i}",
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
            label=f"spin (FCI) {i}",
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


def plot_electron_difference(
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
    for i,j in zip(res_site_index, gen_site_index):
        plt.plot(
            restable[:, 0].astype(complex).real,
            (restable[:, i].astype(complex).real - (gentable[:, j].astype(complex).real + gentable[:, (j + 1)].astype(complex).real)),
        )

    plt.xlabel("Time (au)")
    plt.ylabel("Site Density Error")
    plt.savefig(f"{fig_filename}_error.png")


res_site_index = [1, 2, 3, 4]
gen_site_index = [1, 3, 5, 7]
fig_filename = "electron_density"
plot_two_gen_electron_site_den("gen_h0.dat", "gen_h1.dat", "generalized_density", gen_site_index)
#plot_four_electron_site_den("res_h0.dat", "gen_h0.dat", "res_h1.dat", "gen_h1.dat", fig_filename, res_site_index, gen_site_index)
#plot_multiple_electron_site_den(filename1, filename2, fig_filename, res_site_index, gen_site_index)
#plot_electron_difference(filename1, filename2, fig_filename, res_site_index, gen_site_index)
