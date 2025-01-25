import numpy as np
import matplotlib.pyplot as plt


def make_new_corr_file(old_filename, gen_site_index, res_site_index, res_filename):
    table = []
    openfile = f"{old_filename}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table.append(data)
    table = np.asarray(table)

    restable = []
    openfile = f"{res_filename}"
    with open(openfile, "r") as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            restable.append(data)
    restable = np.asarray(restable)

    new_table = np.empty(restable.shape)
    new_table[:, 0] = np.array(table.astype(complex).real[:, 0])
    for i in gen_site_index:
        # combined_row = np.array(table.astype(complex).real[i][0])
        combined_row = (
            table.astype(complex).real[:, i] + table.astype(complex).real[:, (i + 1)],
        )
        combined_row = np.asarray(combined_row)
        new_table = np.hstack([new_table, combined_row.reshape(-1, 1)])

    # with open(new_filename, "w") as file:
    #    for row in new_table:
    #        file.write("\t".join(str(cell) for cell in row))
    #        file.write("\n")

    plt.figure()
    for i in res_site_index:
        difference = new_table[:, i] - restable.astype(complex).real[:, i]
        plt.scatter(
            new_table[:, 0].astype(complex).real,
            difference,
        )

    plt.xlabel("Time (au)")
    plt.ylabel("Site Density")
    plt.legend()
    plt.savefig(f"error.png")


def plot_multiple_site_den(
    res_filename, gen_filename, fig_filename, res_site_index, gen_site_index, error=True
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

    if error:
        new_table = np.empty(restable.shape)
        new_table[:, 0] = np.array(gentable.astype(complex).real[:, 0])
        for i in gen_site_index:
            combined_row = (
                gentable.astype(complex).real[:, i]
                + gentable.astype(complex).real[:, (i + 1)],
            )
            combined_row = np.asarray(combined_row)
            new_table = np.hstack([new_table, combined_row.reshape(-1, 1)])
            print(new_table)
        plt.figure()
        for i in res_site_index:
            difference = new_table[:, i] - restable.astype(complex).real[:, i]
            print(new_table[:, 1])
            exit()
            plt.scatter(
                new_table[:, 0].astype(complex).real,
                difference,
            )

        plt.xlabel("Time (au)")
        plt.ylabel("Site Density")
        plt.legend()
        plt.savefig("error.png")


res_site_index = [1, 2, 3, 4]
gen_site_index = [1, 3, 5, 7]
genfilename = "electron_density_gen.dat"
resfilename = "electron_density_res.dat"
fig_filename = "site_density.png"
plot_multiple_site_den(
    resfilename, genfilename, fig_filename, res_site_index, gen_site_index, error=False
)
make_new_corr_file(genfilename, gen_site_index, res_site_index, resfilename)
