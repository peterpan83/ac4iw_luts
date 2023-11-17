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

def raleigh():
    s = SixS()
    # s.atmos_profile = AtmosProfile.MidlatitudeSummer
    # s.aero_profile = AeroProfile.NoAerosols
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)


    altitude = Altitudes()
    altitude.set_sensor_custom_altitude(altitude=3)
    altitude.set_target_custom_altitude(0.01)
    s.altitudes = altitude

    # s.wavelength = Wavelength(0.555)
    # s.run()
    # print(s.outputs.reflectance_I)
    # s.outputs.write_output_file("test_out.txt")

    # dic_aerosol_usr = {'soot': 0, 'water': 0, 'oceanic': 0, 'dust': 0}
    # aerosol_components = list(dic_aerosol_usr.keys())
    dic_aerosol_usr = {'soot': 0.25, 'water': 0.25, 'oceanic': 0.25, 'dust': 0.25}
    s.aero_profile = AeroProfile.User(dic_comp=dic_aerosol_usr)
    s.aot550 = 0.1

    reflectance_rayleigh_s, reflectance_aerosol_s, reflectance_total_s = [], [], []

    h2os = [0.005,0.2,1,2,4]
    o3s = [0.1,0.2,0.3,0.4,0.5]
    months = [1,3,6,9,12]
    days = [1,21,22,23,22]
    solar_zs = [0,20,40,60,80]
    viewing_zs = [0,20,40,60,80]
    for i in range(5):
        s.atmos_profile = AtmosProfile.UserWaterAndOzone(water=h2os[i], ozone=o3s[i])
        geo = Geometry.User()
        geo.solar_z = solar_zs[i]
        geo.view_z = viewing_zs[i]
        geo.solar_a = 0
        geo.view_a = 130
        geo.month = months[i]
        geo.day = days[i]
        print(geo.month,geo.day)
        s.geometry = geo
        # s.aero_profile = AeroProfile.User(scoot=0, water=0, oceanic = 1.0, dust=0)

        wv, res = SixSHelpers.Wavelengths.run_wavelengths(s, waves_int * 1e-3)
        reflectance_rayleigh, reflectance_aerosol, reflectance_total = [], [], []
        for re in res:
            reflectance_rayleigh.append(re.reflectance_I.rayleigh)
            reflectance_aerosol.append(re.reflectance_I.aerosol)
            reflectance_total.append(re.reflectance_I.total)


        reflectance_rayleigh_s.append(reflectance_rayleigh)
        reflectance_aerosol_s.append(reflectance_aerosol)

    for i,aero_ref in enumerate(reflectance_rayleigh_s):

        plt.plot(waves_int,aero_ref,'-',label =str([0.005,0.2,1,2,4][i])+','+str([0.1,0.2,0.3,0.4,0.5][i]))

    plt.title('AOT(550)=0.1')
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Rayleigh reflectance')
    plt.show()

raleigh()
