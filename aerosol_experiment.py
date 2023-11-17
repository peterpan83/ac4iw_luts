# dust – The proportion of dust-like aerosols
# water – The proportion of water-like aerosols
# oceanic – The proportion of oceanic aerosols
# soot – The proportion of soot-like aerosols

import numpy as np
from itertools import permutations
from itertools import product
import numpy as np
from Py6S import *
import matplotlib.pyplot as plt

import spectral.io.envi as envi

header = envi.open("190820_MC-C1B-WI-1x1x1_v01-L1G.pix.hdr")
# print(header.metadata)
waves = np.array([float(w) for w in header.metadata['wavelength']])
waves_int = np.array([int(w) for w in waves])

# dust,water,oceanic,soot = np.arange(0,11,1)*0.1,np.arange(0,11,1)*0.1,np.arange(0,11,1)*0.1,np.arange(0,11,1)*0.1

def aerosol_type_combination():
    i = 1
    # for item in permutations(np.arange(0,11,1)*0.1,4):
    #     # if sum(item)==1.0:
    #     #     print(i,item)
    #     #     i+=1
    #     print(i, item)
    #     i += 1
    for item in product(np.arange(0, 11, 1) * 0.1, repeat=4):
        if sum(item) == 1.0:
            print(i, item)
            i += 1


def aerosol_type():
    import spectral.io.envi as envi

    header = envi.open("190820_MC-C1B-WI-1x1x1_v01-L1G.pix.hdr")
    # print(header.metadata)
    waves = np.array([float(w) for w in header.metadata['wavelength']])
    waves_int = np.array([int(w) for w in waves])

    s = SixS()
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water=4.2, ozone=0.3)
    # s.atmos_profile = AtmosProfile.MidlatitudeSummer
    # s.aero_profile = AeroProfile.NoAerosols
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    geo = Geometry.User()
    geo.solar_z = 30
    geo.view_z = 30
    geo.solar_a = 0
    geo.view_a = 130
    s.geometry = geo

    altitude = Altitudes()
    altitude.set_sensor_custom_altitude(altitude=3)
    altitude.set_target_custom_altitude(0.01)
    s.altitudes = altitude

    # s.wavelength = Wavelength(0.555)
    # s.run()
    # print(s.outputs.reflectance_I)
    # s.outputs.write_output_file("test_out.txt")
    reflectance_rayleigh_s, reflectance_aerosol_s, reflectance_total_s = [], [], []
    dic_aerosol_usr = {'soot': 0, 'water': 0, 'oceanic': 0, 'dust': 0}
    aerosol_components = list(dic_aerosol_usr.keys())
    for i in range(4):
        dic_aerosol_usr = {'soot': 0, 'water': 0, 'oceanic': 0, 'dust': 0}
        dic_aerosol_usr[aerosol_components[i]]=1.0
        s.aero_profile = AeroProfile.User(dic_comp=dic_aerosol_usr)
        # s.aero_profile = AeroProfile.User(scoot=0, water=0, oceanic = 1.0, dust=0)
        s.aot550 = 0.1
        wv, res = SixSHelpers.Wavelengths.run_wavelengths(s, waves_int * 1e-3)
        reflectance_rayleigh, reflectance_aerosol, reflectance_total = [], [], []
        for re in res:
            reflectance_rayleigh.append(re.reflectance_I.rayleigh)
            reflectance_aerosol.append(re.reflectance_I.aerosol)
            reflectance_total.append(re.reflectance_I.total)


        reflectance_rayleigh_s.append(reflectance_rayleigh)
        reflectance_aerosol_s.append(reflectance_aerosol)

    for i,aero_ref in enumerate(reflectance_aerosol_s):
        plt.plot(waves_int,aero_ref,'-',label =aerosol_components[i])

    plt.title('AOT(550)=0.1')
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Aerosol reflectance')
    plt.show()

aerosol_type()
