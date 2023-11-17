# generate LUTs (ragleigh,aerosol and gas) for wise image,
## but the program goes more and more slowly when it runs for aerosol simulation that has much more samples
##  to save time, switching to 'gen_lut_inputs_for_wise.py' and 'gen_lut_run.py' is recommended.


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

def genRayleigh(confg,mode):
    waves_6s, waves_centers = getWISEWavelength()

    sz  = parser_parmeter(confg['Geo']['sz'])
    vz = parser_parmeter(confg['Geo']['vz'])
    phi = parser_parmeter(confg['Geo']['phi'])
    f_alt = float(confg['WISE']['altitude'])
    g_alt =  float(confg['Ground']['altitude'])
    sixs_path = confg['6SV']['sixs']
    ncores = int(confg['6SV']['ncores'])

    name = {1:'tropical',2:'midlatsummer',3:'midlatwinter',4:'subarcsummer',5:'subarcwinter'}[mode]

    h5f = tables.open_file('rayleigh_{}_{}_{}.h5'.format(name,confg['WISE']['altitude'],confg['Ground']['altitude']), 'w')
    shape = (len(sz)*len(vz)*len(phi), waves_centers.shape[0])
    atom = Float32Atom()
    atom_int = Int16Atom()
    filters = Filters(complevel=5, complib='zlib')

    h5f.root._v_attrs.Author = 'Yanqun Pan,UQAR,panyq213@163.com'
    h5f.root._v_attrs.Title = 'Raleigh scattering LUT for the WISE sensor'
    ca_wavelength = h5f.create_carray(h5f.root, 'wavelength', atom, waves_centers.shape, filters=filters,title='wavelength')
    ca_sz = h5f.create_carray(h5f.root,'solar_zenith',atom,(len(sz),),filters=filters,title='solar zenith angle (degree)')
    ca_vz = h5f.create_carray(h5f.root, 'viewing_zenith', atom, (len(vz),), filters=filters,title='viewing zenith angle (degree)')
    ca_phi = h5f.create_carray(h5f.root,'relative_azimuth',atom,(len(phi),),filters=filters,title='relative azimuth (degree)')
    ca_rhor =  h5f.create_carray(h5f.root,'raleigh_reflectance',atom,shape,filters=filters,title='rayleigh reflectance')

    ca_spherical_albedo = h5f.create_carray(h5f.root, 'spherical_albedo', atom, shape, filters=filters,title='spherical albedo due to rayleigh scattering')

    ca_transr_up = h5f.create_carray(h5f.root, 'raleigh_trans_up', atom, shape, filters=filters,title='upwelling transmittance due to rayleigh scattering ')
    ca_transr_down = h5f.create_carray(h5f.root, 'raleigh_trans_down', atom, shape, filters=filters,title='downwelling transmittance due to rayleigh scattering')

    # sixs = SixS("/opt/6SV1.1/sixsV1.1")
    if sixs_path == 'None':
        sixs = SixS()
    else:
        sixs = SixS(sixs_path)
    sixs.atmos_profile = AtmosProfile.PredefinedType(mode)

    sixs.aero_profile = AeroProfile.NoAerosols
    sixs.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    geo = Geometry.User()
    geo.day = 20
    geo.month = 8
    geo.solar_a = 0

    altitude = Altitudes()
    altitude.set_sensor_custom_altitude(f_alt)
    altitude.set_target_custom_altitude(g_alt)
    sixs.altitudes = altitude

    rho_r_total,spherical_albedo_total,trans_r_upward_total,trans_r_downward_total  =[], [], [], []
    for isz in tqdm(sz):
        geo.solar_z = isz
        rho_r_vz = []
        trans_r_up_vz = []
        trans_r_down_vz = []
        spherical_albedo_total_vz = []
        for ivz in vz:
            geo.view_z = ivz
            rho_r_phi = []
            trans_r_down_phi = []
            trans_r_up_phi = []
            spherical_albedo_total_phi = []
            for iphi in phi:
                geo.view_a = iphi
                sixs.geometry = geo
                print(isz,ivz,iphi)
                wv, res = SixSHelpers.Wavelengths.run_wavelengths(sixs, waves_6s, n=ncores)
                rho_r = [re.reflectance_I.rayleigh for re in res]
                rho_r_phi.append(rho_r)

                spherical_albedo_r = [re.spherical_albedo.rayleigh for re in res]
                spherical_albedo_total_phi.append(spherical_albedo_r)

                trans_r_upward = [re.transmittance_rayleigh_scattering.upward for re in res]
                trans_r_up_phi.append(trans_r_upward)
                trans_r_downward = [re.transmittance_rayleigh_scattering.downward for re in res]
                trans_r_down_phi.append(trans_r_downward)
            rho_r_vz.append(rho_r_phi)
            trans_r_up_vz.append(trans_r_up_phi)
            trans_r_down_vz.append(trans_r_down_phi)
            spherical_albedo_total_vz.append(spherical_albedo_total_phi)

        rho_r_total.append(rho_r_vz)
        trans_r_downward_total.append(trans_r_down_vz)
        trans_r_upward_total.append(trans_r_up_vz)
        spherical_albedo_total.append(spherical_albedo_total_vz)

    rho_r_total = np.asarray(rho_r_total)
    trans_r_downward_total = np.asarray(trans_r_downward_total)
    trans_r_upward_total = np.asarray(trans_r_upward_total)
    spherical_albedo_total = np.asarray(spherical_albedo_total)

    # r_total = np.concatenate([rho_r_total,trans_r_downward_total,trans_r_upward_total],axis=1)

    ca_wavelength[:] = waves_centers
    ca_sz[:] = np.asarray(sz)
    ca_vz[:] = np.asarray(vz)
    ca_phi[:] = np.asarray(phi)

    ca_rhor[:,:] = rho_r_total.reshape(-1, waves_centers.shape[0])
    ca_transr_up[:,:] = trans_r_upward_total.reshape(-1, waves_centers.shape[0])
    ca_transr_down[:,:] = trans_r_downward_total.reshape(-1, waves_centers.shape[0])
    ca_spherical_albedo[:,:] = spherical_albedo_total.reshape(-1,waves_centers.shape[0])
    h5f.close()

    # np.savetxt('rhor_{}.txt'.format(imagename), rho_r_total.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    # np.savetxt('transr_up_{}.txt'.format(imagename), trans_r_upward_total.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    # np.savetxt('transr_down_{}.txt'.format(imagename), trans_r_downward_total.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    #  (1, 4, 2, 144)
    return rho_r_total,ca_spherical_albedo,trans_r_downward_total,trans_r_upward_total


def genAerosol(confg):
    i = 1
    aero_coms = []
    for item in product(np.arange(0, 12, 2) * 0.1, repeat=4):
        if sum(item) == 1.0 and item[3] < 0.6 and item[0] < 0.6:
            print(i, item)
            aero_coms.append({'soot': item[0], 'water': item[1], 'oceanic': item[2], 'dust': item[3]})
            i += 1
    start_i, end_i =int(confg['Aerosol']['start_i']), int(confg['Aerosol']['end_i'])
    aero_coms = aero_coms[start_i:end_i]

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
        atom = Float32Atom()
        atom_int = Int16Atom()
        filters = Filters(complevel=5, complib='zlib')

        h5f = tables.open_file(os.path.join(out_dir,output_name),'w',title='Aerosol scattering LUT for the WISE sensor')
        h5f.root._v_attrs.Author = 'Yanqun Pan,UQAR,panyq213@163.com'
        h5f.root._v_attrs.AerosolType = str(aero_coms[key])
        # h5f.root._v_attrs.Title = 'Aerosol scattering LUT for the WISE sensor'
        ca_wavelength = h5f.create_carray(h5f.root, 'wavelength', atom, waves_centers.shape, filters=filters,
                                          title='wavelength')
        ca_sz = h5f.create_carray(h5f.root, 'solar_zenith', atom, (len(sz),), filters=filters,
                                  title='solar zenith angle (degree)')
        ca_vz = h5f.create_carray(h5f.root, 'viewing_zenith', atom, (len(vz),), filters=filters,
                                  title='viewing zenith angle (degree)')
        ca_phi = h5f.create_carray(h5f.root, 'relative_azimuth', atom, (len(phi),), filters=filters,
                                   title='relative azimuth (degree)')
        ca_taua550 = h5f.create_carray(h5f.root, 'taua_550', atom, (len(taua550),), filters=filters,
                                   title='relative azimuth (degree)')

        ca_rhoa = h5f.create_carray(h5f.root, 'aerosol_reflectance', atom, shape, filters=filters,
                                    title='aerosol reflectance')
        ca_transa_up = h5f.create_carray(h5f.root, 'aerosol_trans_up', atom, shape, filters=filters,
                                         title='upwelling transmittance due to aerosol scattering ')
        ca_transa_down = h5f.create_carray(h5f.root, 'aerosol_trans_down', atom, shape, filters=filters,
                                           title='downwelling transmittance due to aerosol scattering')

        ca_wavelength[:] = waves_centers
        ca_sz[:] = np.asarray(sz)
        ca_vz[:] = np.asarray(vz)
        ca_phi[:] = np.asarray(phi)
        ca_taua550[:] = np.asarray(taua550)

        paras = paras_dic[key]
        rhoa,trans_a_up,trans_a_down = [],[],[]
        for k,para in tqdm(enumerate(paras)):
            tua, isz, ivz, iphi =  para
            geo.solar_z = isz
            geo.view_z = ivz
            geo.view_a = iphi
            sixs.geometry = geo

            sixs.aot550 = tua
            wv, res = SixSHelpers.Wavelengths.run_wavelengths_single(sixs, waves_6s, n=ncores)

            rho_a = [re.reflectance_I.aerosol for re in res]
            rhoa.append(rho_a)

            trans_a_upward = [re.transmittance_aerosol_scattering.upward for re in res]
            trans_a_up.append(trans_a_upward)

            trans_a_downward = [re.transmittance_aerosol_scattering.downward for re in res]
            trans_a_down.append(trans_a_downward)
            # ca_rhoa[k, :] = rho_a
            # ca_transa_up[k, :] = trans_a_upward
            # ca_transa_down[k, :] = trans_a_downward
            # trans_a_down.append(trans_a_downward)

        rhoa = np.asarray(rhoa)
        trans_a_up = np.asarray(trans_a_up)
        trans_a_down = np.asarray(trans_a_down)

        ca_rhoa[:, :] = rhoa.reshape(-1, waves_centers.shape[0])
        ca_transa_up[:, :] = trans_a_up.reshape(-1, waves_centers.shape[0])
        ca_transa_down[:, :] = trans_a_down.reshape(-1, waves_centers.shape[0])

        h5f.close()

    return 0

def genGasTrans(confg):
    waves_6s, waves_centers = getWISEWavelength()

    sz = parser_parmeter(confg['Geo']['sz'])
    vz = parser_parmeter(confg['Geo']['vz'])
    phi = parser_parmeter(confg['Geo']['phi'])
    wv, o3 = parser_parmeter(confg['Gas']['wv']),parser_parmeter(confg['Gas']['o3'])

    f_alt = float(confg['WISE']['altitude'])
    g_alt = float(confg['Ground']['altitude'])

    sixs_path = confg['6SV']['sixs']
    ncores = int(confg['6SV']['ncores'])
    out_dir = confg['IO']['out_dir']

    sixs = SixS(sixs_path)
    sixs.aero_profile = AeroProfile.NoAerosols
    sixs.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    geo = Geometry.User()
    geo.day = 20
    geo.month = 8
    geo.solar_a = 0

    altitude = Altitudes()
    altitude.set_sensor_custom_altitude(f_alt)
    altitude.set_target_custom_altitude(g_alt)
    sixs.altitudes = altitude

    output_name = "gas_absorption_3_0.h5"
    shape_down = (len(sz) * len(o3)*len(wv), waves_centers.shape[0])
    shape_up = (len(vz) * len(o3)*len(wv), waves_centers.shape[0])
    atom = Float32Atom()
    filters = Filters(complevel=5, complib='zlib')

    h5f = tables.open_file(os.path.join(out_dir, output_name), 'w', title='Gas absorption transmittance LUT for the WISE sensor')
    h5f.root._v_attrs.Author = 'Yanqun Pan,UQAR,panyq213@163.com'
    h5f.root._v_attrs.DimensionOrder = '(sz,vz,phi,o3,wv)'
    # h5f.root._v_attrs.Title = 'Aerosol scattering LUT for the WISE sensor'
    ca_wavelength = h5f.create_carray(h5f.root, 'wavelength', atom, waves_centers.shape, filters=filters,
                                      title='wavelength')
    ca_sz = h5f.create_carray(h5f.root, 'solar_zenith', atom, (len(sz),), filters=filters,
                              title='solar zenith angle (degree)')
    ca_vz = h5f.create_carray(h5f.root, 'viewing_zenith', atom, (len(vz),), filters=filters,
                              title='viewing zenith angle (degree)')
    ca_phi = h5f.create_carray(h5f.root, 'relative_azimuth', atom, (len(phi),), filters=filters,
                               title='relative azimuth (degree)')
    ca_o3 = h5f.create_carray(h5f.root, 'ozone', atom, (len(o3),), filters=filters,
                                   title='total ozone concentration(atm.cm)')
    ca_wv = h5f.create_carray(h5f.root, 'water_vapor', atom, (len(wv),), filters=filters,
                                   title='total water vapor volumn (g/cm^2)')

    ca_transg_up = h5f.create_carray(h5f.root, 'gas_trans_up', atom, shape_up, filters=filters,
                                     title='upwelling transmittance due to gas absorption ')
    ca_transg_down = h5f.create_carray(h5f.root, 'gas_trans_down', atom, shape_down, filters=filters,
                                       title='downwelling transmittance due to gas absorption')

    ca_wavelength[:] = waves_centers
    ca_sz[:] = np.asarray(sz)
    ca_vz[:] = np.asarray(vz)
    ca_phi[:] = np.asarray(phi)
    ca_o3[:]  =  np.asarray(o3)
    ca_wv[:] = np.asarray(wv)

    parameters_down,parameters_up = [],[]
    for isz in sz:
        for io3 in o3:
            for iwv in wv:
                parameters_down.append((isz,io3,iwv))

    for ivz in vz:
        for io3 in o3:
            for iwv in wv:
                parameters_up.append((ivz,io3,iwv))

    trans_up,trans_down = [], []
    for p in tqdm(np.asarray(parameters_down)):
        isz, io3, iwv = p
        sixs.atmos_profile = AtmosProfile.UserWaterAndOzone(water=iwv, ozone=io3)
        geo.solar_z = isz
        geo.view_z = 30
        geo.view_a = 40
        sixs.geometry = geo
        _, res = SixSHelpers.Wavelengths.run_wavelengths(sixs, waves_6s, n=ncores)

        trans_g_downward = [re.transmittance_global_gas.downward for re in res]
        trans_down.append(trans_g_downward)
    ca_transg_down[:,:] = np.asarray(trans_down)
    del trans_down

    for p in tqdm(np.asarray(parameters_up)):
        ivz, io3, iwv = p
        sixs.atmos_profile = AtmosProfile.UserWaterAndOzone(water=iwv, ozone=io3)
        geo.solar_z = 30
        geo.view_z = ivz
        geo.view_a = 40
        sixs.geometry = geo
        _, res = SixSHelpers.Wavelengths.run_wavelengths(sixs, waves_6s, n=ncores)

        trans_g_upward = [re.transmittance_global_gas.upward for re in res]
        trans_up.append(trans_g_upward)
    ca_transg_up[:,:] = np.asarray(trans_up)

    h5f.close()



if __name__ == '__main__':
    config_f = sys.argv[1]
    simulation = sys.argv[2]

    confg = ConfigParser()
    confg.read(config_f)
    imagename = confg['WISE']['imagename']
    # getWISEWavelength(imagename)

    if simulation == 'rayleigh':
        print("# Tropical = 1,MidlatitudeSummer = 2,MidlatitudeWinter = 3,SubarcticSummer = 4,SubarcticWinter = 5")
        mode = int(sys.argv[3])
        rho, albedo,tu, td = genRayleigh(confg,mode)
    elif simulation == 'aerosol':
        genAerosol(confg)
    elif simulation == 'gas':
        genGasTrans(confg)





    # waves_int = np.array([int(w) for w in waves])


    # rho, tu, td = genAerosol(confg, waves_int)
    # np.savetxt('rhoa_{}.txt'.format(imagename), rho.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    # np.savetxt('transa_up_{}.txt'.format(imagename), tu.reshape(-1, 144), delimiter=',', fmt='%1.4e')
    # np.savetxt('transa_down_{}.txt'.format(imagename), td.reshape(-1, 144), delimiter=',', fmt='%1.4e')

    # rho, tu, td = genRayleigh(confg, waves_int)

    # s_res = genGasTrans(confg, waves_int)







