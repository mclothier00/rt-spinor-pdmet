import numpy as np
import matplotlib.pyplot as plt

# site index starting from 0

def plot_site_den(filename, site_index):
    table = []
    openfile = F'{filename}'

    with open(openfile, 'r') as f:
        for line in f:
            data = line.split()
            data = [x.strip() for x in data]
            table.append(data)

    table = np.asarray(table)

    plt.figure()
    for i in site_index:
        plt.plot(table[:,0].astype(complex).real, table[:,i].astype(complex).real)
    plt.xlabel('Time (au)')
    plt.ylabel('Site Density')
    plt.savefig(F'site_density_tdhf.png')

site_index = [3]
filename = 'corr_density.dat'
plot_site_den(filename, site_index)
