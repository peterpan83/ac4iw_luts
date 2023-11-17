from Py6S import *
import numpy as np
import os

out_dir_ = os.path.join(os.path.split(os.path.abspath(__file__))[0],"inputs")

waves_wise = [361.51,365.93,370.36,374.78,379.21,383.63,388.05,392.48,396.9,401.32,405.74,410.16,414.58,419.0,423.42,427.84,432.26,436.68,441.1,445.52,449.93,454.35,458.77,463.18,467.6,472.01,476.43,480.84,485.25,489.67,494.08,498.49,502.9,507.32,511.73,516.14,520.55,524.96,529.37,533.78,538.19,542.6,547.0,551.41,555.82,560.23,564.63,569.04,573.45,577.85,582.26,586.66,591.07,595.47,599.88,604.28,608.69,613.09,617.49,621.9,626.3,630.7,635.1,639.5,643.91,648.31,652.71,657.11,661.51,665.91,670.31,674.71,679.11,683.51,687.9,692.3,696.7,701.1,705.5,709.89,714.29,718.69,723.09,727.48,731.88,736.28,740.67,745.07,749.46,753.86,758.25,762.65,767.04,771.44,775.83,780.23,784.62,789.01,793.41,797.8,802.19,806.59,810.98,815.37,819.77,824.16,828.55,832.94,837.34,841.73,846.12,850.51,854.9,859.3,863.69,868.08,872.47,876.86,881.25,885.64,890.03,894.42,898.81,903.21,907.6,911.99,916.38,920.77,925.16,929.55,933.94,938.33,942.72,947.11,951.5,955.89,960.28,964.67,969.06,973.45,977.83,982.22,986.61,991.0]

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

sixs = SixS()

waves_6s, waves_centers = getWISEWavelength()
sixs.atmos_profile = AtmosProfile.SubarcticSummer

# 0.13 {'soot': 0.30000000000000004, 'water': 0.0, 'oceanic': 0.2, 'dust': 0.5}
# sixs.aero_profile = AeroProfile.User(**{'soot': 0.3, 'water': 0., 'oceanic': 0.2, 'dust': 0.5})
sixs.aero_profile = AeroProfile.Stratospheric
sixs.aot550 = 0.5
sixs.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

geo = Geometry.User()
geo.day = 20
geo.month = 8

# 'SZA':36.6,'SAA':165,'VZA':2.685,'VAA':113.3
geo.solar_a = 165

altitude = Altitudes()
altitude.set_sensor_custom_altitude(3.0)
# altitude.set_sensor_satellite_level()
altitude.set_target_custom_altitude(0)
sixs.altitudes = altitude
geo.solar_z = 36.6
# geo.solar_z = 0
geo.view_z = 2.685
geo.view_a =113.3
sixs.geometry = geo
SixSHelpers.Wavelengths.run_wavelenght_single_inputs(sixs,waves_6s,out_dir_,1)
