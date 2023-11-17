import numpy as np
import scipy.interpolate as interp
from configparser import ConfigParser
import spectral.io.envi as envi

import matplotlib.pyplot as plt


confg = ConfigParser()
confg.read('config_wise_luts')
imagename = confg['WISE']['imagename']

header = envi.open("{}-L1G.pix.hdr".format(imagename))
# print(header.metadata)
waves = np.array([float(w) for w in header.metadata['wavelength']])
waves_int = np.array([int(w) for w in waves])
print(','.join([str(w) for w in waves_int]))

def fillNA(name):
    data = np.loadtxt('{}_190820_MC-C1B-WI-1x1x1_v01.txt'.format(name),delimiter=',')
    lines = []
    for item in data:
        mask = item>0
        v = [str(x) for x in list(interp.interp1d(waves_int[mask],item[mask])(waves_int))]
        lines.append(','.join(v)+'\n')
    print(lines)
    with open('{}.txt'.format(name),'w') as f:
        f.writelines(lines)


def load(name):
    a = np.loadtxt('{}.txt'.format(name), delimiter=',')
    print(a.shape)
    for i in range(a.shape[0]):
        plt.plot(waves_int[:],a[i][:])

    # plt.ylim(0,0.005)
    plt.show()

name='rhoa'
# fillNA(name=name)
load(name=name)



