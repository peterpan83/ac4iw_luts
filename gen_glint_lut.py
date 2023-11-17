import os,sys
import tables
from tables import *
from tables import Float32Atom
from tables import Int32Atom
from tables import Filters
import numpy as np
from tqdm import tqdm
from configparser import ConfigParser

def parser_parmeter(config_value):
    return [float(v) for v in config_value.split(',')]

def praser_output(ofile):
    with open(ofile,'r') as f:
        line = f.readline()
    line = line.strip()
    data = line.split(",")
    nLg = float(data[-1])
    # os.remove(ofile)
    return nLg


def get_glint(confg):
    sz  = parser_parmeter(confg['Geo']['sz'])
    vz = parser_parmeter(confg['Geo']['vz'])
    phi = parser_parmeter(confg['Geo']['phi'])
    windspeed = parser_parmeter(confg['METEO']['windspeed'])
    out_dir = confg['IO']['out_dir']
    output_name = 'glint_Cox_Munk.h5'

    shape = (len(sz) * len(vz) * len(phi),len(windspeed))
    atom = Float32Atom()
    filters = Filters(complevel=5, complib='zlib')

    h5f = tables.open_file(os.path.join(out_dir, output_name), 'w',
                           title='glint coefficient based on Cox and Munk model')
    h5f.root._v_attrs.Author = 'Yanqun Pan,UQAR,panyq213@163.com'
    h5f.root._v_attrs.DimensionOrder = '(sz,vz,phi,windspeed)'
    # h5f.root._v_attrs.Title = 'Aerosol scattering LUT for the WISE sensor'

    ca_sz = h5f.create_carray(h5f.root, 'solar_zenith', atom, (len(sz),), filters=filters,
                              title='solar zenith angle (degree)')
    ca_vz = h5f.create_carray(h5f.root, 'viewing_zenith', atom, (len(vz),), filters=filters,
                              title='viewing zenith angle (degree)')
    ca_phi = h5f.create_carray(h5f.root, 'relative_azimuth', atom, (len(phi),), filters=filters,
                               title='relative azimuth (degree)')
    ca_ws = h5f.create_carray(h5f.root, 'wind_speed', atom, (len(windspeed),), filters=filters,
                              title='wind speed (m/s)')

    ca_sz[:] = np.asarray(sz)
    ca_vz[:] = np.asarray(vz)
    ca_phi[:] = np.asarray(phi)
    ca_ws[:] = np.asarray(windspeed)
    ca_glint = h5f.create_carray(h5f.root, 'nLg', atom, shape, filters=filters,
                                title='glint coefficient')

    parameters = []
    for sz_ in sz:
        for vz_ in vz:
            for phi_ in phi:
                for ws_ in windspeed:
                    parameters.append([sz_,vz_,phi_,ws_])

    i = 0
    nLg_s = []
    for p in tqdm(parameters[:]):
        out_f = "/mnt/d/Work/Programe/WISE_LUTs/glint/glint_%d.txt"%(i)
        if not os.path.exists(out_f):
            os.system('perl ./glint/example.pl %s %.3f %.3f %.3f %.3f >%s' % (str(i), p[1],p[0], p[2], p[3],out_f))
        nLg_s.append(praser_output(out_f))
        i+=1

    nLg_s = np.asarray(nLg_s)
    nLg_s_ = nLg_s.reshape(len(sz)*len(vz)*len(phi),len(windspeed))
    ca_glint[:,:]=nLg_s_
    h5f.close()









if __name__ == '__main__':
    config_f = sys.argv[1]
    confg = ConfigParser()
    confg.read(config_f)
    get_glint(confg)
    # os.system('perl example.pl %s %.3f %.3f %.3f %.3f'%('0', 15.000, 1.000, 0.000, 0.500))
