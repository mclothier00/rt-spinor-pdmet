import numpy as np
import matplotlib.pyplot as plt

# site index starting from 0


def plot_multiple_site_den(
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


res_site_index = [1, 2]
gen_site_index = [1, 3]
genfilename = "electron_density_gen.dat"
resfilename = "electron_density_res.dat"
fig_filename = "site_density.png"
plot_multiple_site_den(
    resfilename, genfilename, fig_filename, res_site_index, gen_site_index
)
