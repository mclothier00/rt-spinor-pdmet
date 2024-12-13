import numpy as np
import matplotlib.pyplot as plt

# site index starting from 0


def plot_site_den(filename, fig_filename, site_index, gen=False):
    table = []
    openfile = f"{filename}"

    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table.append(data)

    table = np.asarray(table)

    plt.figure()
    if not gen:
        for i in site_index:
            plt.plot(table[:, 0].astype(complex).real, table[:, i].astype(complex).real)
    if gen:
        for i in site_index:
            site_den = (
                table[:, i].astype(complex).real
                + table[:, (i + 1)].astype(complex).real
            )
            plt.plot(table[:, 0].astype(complex).real, site_den)

    plt.xlabel("Time (au)")
    plt.ylabel("Site Density")
    plt.savefig(f"{fig_filename}")


site_index = [1, 3, 5, 7]
filename = "electron_density.dat"
fig_filename = "site_density.png"
plot_site_den(filename, fig_filename, site_index)
