# generate inputs file for 6SV1.1
# for aerosol simuation

import sys,os
from tqdm import tqdm
import numpy as np
from itertools import product
from configparser import ConfigParser
import tables
from tables import *
from tables import Float32Atom
from tables import Int32Atom
from tables import Filters

from Py6S import *


waves_wise = [361.51,365.93,370.36,374.78,379.21,383.63,388.05,392.48,396.9,401.32,405.74,410.16,414.58,419.0,423.42,427.84,432.26,436.68,441.1,445.52,449.93,454.35,458.77,463.18,467.6,472.01,476.43,480.84,485.25,489.67,494.08,498.49,502.9,507.32,511.73,516.14,520.55,524.96,529.37,533.78,538.19,542.6,547.0,551.41,555.82,560.23,564.63,569.04,573.45,577.85,582.26,586.66,591.07,595.47,599.88,604.28,608.69,613.09,617.49,621.9,626.3,630.7,635.1,639.5,643.91,648.31,652.71,657.11,661.51,665.91,670.31,674.71,679.11,683.51,687.9,692.3,696.7,701.1,705.5,709.89,714.29,718.69,723.09,727.48,731.88,736.28,740.67,745.07,749.46,753.86,758.25,762.65,767.04,771.44,775.83,780.23,784.62,789.01,793.41,797.8,802.19,806.59,810.98,815.37,819.77,824.16,828.55,832.94,837.34,841.73,846.12,850.51,854.9,859.3,863.69,868.08,872.47,876.86,881.25,885.64,890.03,894.42,898.81,903.21,907.6,911.99,916.38,920.77,925.16,929.55,933.94,938.33,942.72,947.11,951.5,955.89,960.28,964.67,969.06,973.45,977.83,982.22,986.61,991.0]

def FWMH2RSR(FWMHs,centers,resolution,wrange=(200,2000),model='gaussian'):
    '''
    convert FWMH and center wavelenth to RSR
    :param FWMHs: 1D ndarray,  full width at half maximum
    :param centers: 1D ndarray, center wavelength
    :param resolution: the spectral resolution for intergration
    :param wrange: the range of the wavelength for intergration
    :param model: model for simulation of RSR,the default is Gaussian
    :return: list of rsr ordered by bands,wavelenghs
    '''
    x = np.arange(wrange[0],wrange[1],resolution)
    if model=='gaussian':
        ##let's use Gaussian funcrion f=a*exp[-(x-b)^2/2c^2] to simulate RSR (relative spectral response)
        ##since Maximum of RSR is 1, a is set to 1.
        ## references:
        ### https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        ### http://blog.sina.com.cn/s/blog_4a1c6f7f0100061m.html
        cs = [fwmh/2.0/np.sqrt(2*np.log(2)) for fwmh in FWMHs]
        rsr = [np.exp(-np.power(x-b,2)/(2*c**2)) for c,b in zip(cs,centers)]
    else:
        return None
    return rsr,x

def getWISEWavelength():
    import matplotlib.pyplot as plt
    # header = envi.open("{}-L1G.pix.hdr".format(imagename))
    # print(header.metadata)
    # centers = np.array([float(w) for w in header.metadata['wavelength']])
    centers = np.array(waves_wise)
    print(','.join([str(center) for center in centers]))
    nbands = 144
    FWMHs = np.full(nbands, 5.05)

    wavelength_6s = []
    for i,center in enumerate(centers):
        x = np.array([center-i*2.5 for i in reversed(range(6))]+[center+i*2.5 for i in range(1,6)])
        # cs = [fwmh/2.0/np.sqrt(2*np.log(2)) for fwmh in FWMHs]
        cs =  FWMHs[i]/2.0/np.sqrt(2*np.log(2))
        rsr = np.exp(-np.power(x-center,2)/(2*cs**2))
        # plt.plot(x,rsr)
        # plt.show()
        wavelength_6s.append((139+i,x[0]*1e-3,x[-1]*1e-3,rsr))

    return wavelength_6s,centers


def parser_parmeter(config_value):
    return [float(v) for v in config_value.split(',')]

def genUsrDefineAerosolType():
    i = 1
    aero_coms = []
    for item in product(np.arange(0, 12, 2) * 0.1, repeat=4):
        if sum(item) == 1.0 and item[3] < 0.6 and item[0] < 0.6:
            print(i, item)
            aero_coms.append({'soot': item[0], 'water': item[1], 'oceanic': item[2], 'dust': item[3]})
            i += 1

    return aero_coms

def genAerosol(confg):
    waves_6s, waves_centers = getWISEWavelength()
    sz = parser_parmeter(confg['Geo']['sz'])
    vz = parser_parmeter(confg['Geo']['vz'])
    phi = parser_parmeter(confg['Geo']['phi'])
    sixs_path = confg['6SV']['sixs']
    ncores = int(confg['6SV']['ncores'])
    out_dir = confg['IO']['out_dir']

    taua550 = parser_parmeter(confg['Aerosol']['taua550'])

    f_alt = float(confg['WISE']['altitude'])
    g_alt = float(confg['Ground']['altitude'])

    sixs = SixS() if sixs_path == 'None' else SixS(sixs_path)

    sixs.atmos_profile = AtmosProfile.MidlatitudeSummer
    sixs.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    geo = Geometry.User()
    geo.day = 20
    geo.month = 8
    geo.solar_a = 0

    altitude = Altitudes()
    altitude.set_sensor_custom_altitude(f_alt)
    altitude.set_target_custom_altitude(g_alt)
    sixs.altitudes = altitude

    start_i, end_i = int(confg['Aerosol']['start_i']), int(confg['Aerosol']['end_i'])
    aero_coms = genUsrDefineAerosolType()
    aero_coms = aero_coms[start_i:end_i]
    paras_dic = {}
    for i, a_com in enumerate(aero_coms):
        parameters = []
        for tua in taua550:
            for isz in sz:
                for ive in vz:
                    for iphi in phi:
                        parameters.append((tua, isz, ive, iphi))

        paras_dic[i] = parameters

    for key in paras_dic.keys():
        sixs.aero_profile = AeroProfile.User(aero_coms[key])

        output_name = 'aerosol_midlatsummer_{}_{}_{}.h5'.format(confg['WISE']['altitude'], confg['Ground']['altitude'],key+start_i)

        shape = (len(sz) * len(vz) * len(phi)*len(taua550), waves_centers.shape[0])

        paras = paras_dic[key]
        out_dir_ = os.path.join(out_dir,'aerosol_{}'.format(key+start_i))
        if not os.path.exists(out_dir_):
            os.mkdir(out_dir_)
        for k,para in tqdm(enumerate(paras)):
            tua, isz, ivz, iphi =  para
            geo.solar_z = isz
            geo.view_z = ivz
            geo.view_a = iphi
            sixs.geometry = geo

            sixs.aot550 = tua
            SixSHelpers.Wavelengths.run_wavelenght_single_inputs(sixs,waves_6s,out_dir_,k)

            # ca_rhoa[k, :] = rho_a
            # ca_transa_up[k, :] = trans_a_upward
            # ca_transa_down[k, :] = trans_a_downward
            # trans_a_down.append(trans_a_downward)
    return 0


def genAerosol_test(confg):
    waves_6s, waves_centers = getWISEWavelength()
    sz = parser_parmeter(confg['Geo']['sz'])
    vz = parser_parmeter(confg['Geo']['vz'])
    phi = parser_parmeter(confg['Geo']['phi'])
    sixs_path = confg['6SV']['sixs']
    ncores = int(confg['6SV']['ncores'])
    out_dir = confg['IO']['out_dir']

    taua550 = parser_parmeter(confg['Aerosol']['taua550'])

    f_alt = float(confg['WISE']['altitude'])
    g_alt = float(confg['Ground']['altitude'])

    sixs = SixS() if sixs_path == 'None' else SixS(sixs_path)

    sixs.atmos_profile = AtmosProfile.MidlatitudeSummer
    sixs.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    geo = Geometry.User()
    geo.day = 20
    geo.month = 8
    geo.solar_a = 0

    altitude = Altitudes()
    altitude.set_sensor_custom_altitude(f_alt)
    altitude.set_target_custom_altitude(g_alt)
    sixs.altitudes = altitude

    start_i, end_i = int(confg['Aerosol']['start_i']), int(confg['Aerosol']['end_i'])
    aero_coms = genUsrDefineAerosolType()
    aero_coms = aero_coms[start_i:end_i]
    paras_dic = {}
    for i, a_com in enumerate(aero_coms):
        parameters = []
        for tua in taua550:
            for isz in sz:
                for ive in vz:
                    for iphi in phi:
                        parameters.append((tua, isz, ive, iphi))

        paras_dic[i] = parameters

    for key in paras_dic.keys():
        sixs.aero_profile = AeroProfile.User(aero_coms[key])

        output_name = 'aerosol_midlatsummer_{}_{}_{}.h5'.format(confg['WISE']['altitude'], confg['Ground']['altitude'],key+start_i)

        shape = (len(sz) * len(vz) * len(phi)*len(taua550), waves_centers.shape[0])

        paras = paras_dic[key]
        out_dir_ = os.path.join(out_dir,'aerosol_{}'.format(key+start_i))
        if not os.path.exists(out_dir_):
            os.mkdir(out_dir_)
        for k,para in tqdm(enumerate(paras)):
            tua, isz, ivz, iphi =  para
            geo.solar_z = isz
            geo.view_z = ivz
            geo.view_a = iphi
            sixs.geometry = geo

            sixs.aot550 = tua
            SixSHelpers.Wavelengths.run_wavelenght_single_inputs(sixs,waves_6s,out_dir_,k)

            # ca_rhoa[k, :] = rho_a
            # ca_transa_up[k, :] = trans_a_upward
            # ca_transa_down[k, :] = trans_a_downward
            # trans_a_down.append(trans_a_downward)
    return 0


if __name__ == '__main__':

    config_f = sys.argv[1]
    simulation = sys.argv[2]
    confg = ConfigParser()
    confg.read(config_f)
    imagename = confg['WISE']['imagename']
    # getWISEWavelength(imagename)

    if simulation == 'aerosol':
        genAerosol(confg)





    # waves_int = np.array([int(w) for w in waves])


    # rho, tu, td = genAerosol(confg, waves_int)
    # np.savetxt('rhoa_{}.txt'.format(imagename), rho.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    # np.savetxt('transa_up_{}.txt'.format(imagename), tu.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    # np.savetxt('transa_down_{}.txt'.format(imagename), td.reshape(-1, 144), delimiter=',', fmt='%1.4e')

    # rho, tu, td = genRayleigh(confg, waves_int)

    # s_res = genGasTrans(confg, waves_int)







