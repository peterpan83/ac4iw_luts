import h5py,os,sys
import numpy as np
import math

data_dir = 'D:\Work\Programe\WISE_LUTs\path\output'
# data_dir = '/mnt/d/Work/Programe/WISE_LUTs/path/output'
# def fillNA4Aerosol(aerosol_type):
#     lut_path = os.path.join(data_dir,'aerosol_midlatsummer_3_0_{}.h5'.format(aerosol_type))
#     data = h5py.File(lut_path,'r+')
#
#     for item in ['aerosol_reflectance','aerosol_trans_up','aerosol_trans_down','spherical_albedo']:
#         data_d = data[item].value
#         # print(data_d[:,73])
#         # print(type(data_d[:, 73][0]),math.isnan(data_d[:, 73][0]))
#         data[item][:,73] = (data_d[:,72]+data_d[:,74])*0.5
#         # print(data_d.dtype)
#
#
#         data_d[:,73] = (data_d[:,72]+data_d[:,74])*0.5
#         print(np.where(np.isnan(data_d)))
#         # mask = ~(data_d>0)
#         rows,cols =  np.where(np.isnan(data_d))
#         # rows, cols = np.where(mask==True)
#         for row,col in zip(rows,cols):
#             up = None if row-1<0 else data_d[row-1,col]
#             down = None if row+1>=data_d.shape[0] else data_d[row+1,col]
#             left = None if col-1<0 else data_d[row,col-1]
#             right = None if col+1>data_d.shape[1] else data_d[row,col+1]
#
#             if data_d[row-1][col] == data_d[row+1][col]:
#                 fill_value =  data_d[row+1][col]
#             elif data_d[row][col-1] == data_d[row][col+1]:
#                 fill_value =  data_d[row][col+1]
#             else:
#                 scale_up_row = (data_d[row-1][col-1]+data_d[row-1][col+1])/data_d[row-1][col]
#                 scale_down_row = (data_d[row +1][col - 1] + data_d[row +1][col + 1]) / data_d[row + 1][col]
#
#                 scale =  (scale_up_row+ scale_down_row)/2
#                 fill_value = (data_d[row][col-1]+data_d[row][col+1])/scale
#
#             data[item][row, col] = fill_value
#
#     data.close()


def fillNA_path(lut='rayleigh',aerosol_type=None):
    '''
    :param lut: rayleigh or gas or aerosol, if aerosol,aerosol_type should not be None
    :return:
    '''


    lut_path = os.path.join(data_dir, '{}_3_0_{}.h5'.format(lut,aerosol_type))
    dataset_names = ['diffuse_reflectance','diffuse_irradiance','path_radiance','path_reflectance',
                     'path_t_reflectance',  'trans_up', 'trans_down', 'spherical_albedo']

    print(lut_path)

    data = h5py.File(lut_path,'r+')

    for item in dataset_names:
        data_d = data[item].value

        for i in range(data_d.shape[1]):
            if np.where(np.isnan(data_d[:,i]))[0].shape[0]==data_d.shape[0]:
                data[item][:, i] = (data_d[:, i-1] + data_d[:, i+1]) * 0.5
                data_d[:, i] = (data_d[:, i-1] + data_d[:, i+1]) * 0.5

        print(np.where(np.isnan(data_d)))
        # mask = ~(data_d>0)
        rows,cols =  np.where(np.isnan(data_d))
        # rows, cols = np.where(mask==True)
        for row,col in zip(rows,cols):
            up = None if row-1<0 else data_d[row-1,col]
            down = None if row+1>=data_d.shape[0] else data_d[row+1,col]
            left = None if col-1<0 else data_d[row,col-1]
            right = None if col+1>data_d.shape[1] else data_d[row,col+1]

            if data_d[row-1][col] == data_d[row+1][col]:
                fill_value =  data_d[row+1][col]
            elif data_d[row][col-1] == data_d[row][col+1]:
                fill_value =  data_d[row][col+1]
            else:
                scale_up_row = (data_d[row-1][col-1]+data_d[row-1][col+1])/data_d[row-1][col]
                scale_down_row = (data_d[row +1][col - 1] + data_d[row +1][col + 1]) / data_d[row + 1][col]

                scale =  (scale_up_row+ scale_down_row)/2
                fill_value = (data_d[row][col-1]+data_d[row][col+1])/scale

            data[item][row, col] = fill_value

    data.close()

def fillNA(lut='rayleigh',aerosol_type=None,atmosphere_mode=None):
    '''
    :param lut: rayleigh or gas or aerosol, if aerosol,aerosol_type should not be None
    :return:
    '''

    if lut== 'rayleigh':
        lut_path = os.path.join(data_dir,'{}_{}_3_0.h5'.format(lut,atmosphere_mode))
        dataset_names = ['raleigh_reflectance', 'raleigh_trans_up', 'raleigh_trans_down', 'spherical_albedo']
    elif lut=='gas':
        lut_path = os.path.join(data_dir, '{}_absorption_3_0.h5'.format(lut))
        dataset_names = ['gas_trans_down', 'gas_trans_up']
    else:
        lut_path = os.path.join(data_dir, '{}_midlatsummer_3_0_{}.h5'.format(lut,aerosol_type))
        dataset_names = ['aerosol_reflectance', 'aerosol_trans_up', 'aerosol_trans_down', 'spherical_albedo']

    print(lut_path)

    data = h5py.File(lut_path,'r+')

    for item in dataset_names:
        data_d = data[item].value

        for i in range(data_d.shape[1]):
            if np.where(np.isnan(data_d[:,i]))[0].shape[0]==data_d.shape[0]:
                data[item][:, i] = (data_d[:, i-1] + data_d[:, i+1]) * 0.5
                data_d[:, i] = (data_d[:, i-1] + data_d[:, i+1]) * 0.5

        print(np.where(np.isnan(data_d)))
        # mask = ~(data_d>0)
        rows,cols =  np.where(np.isnan(data_d))
        # rows, cols = np.where(mask==True)
        for row,col in zip(rows,cols):
            up = None if row-1<0 else data_d[row-1,col]
            down = None if row+1>=data_d.shape[0] else data_d[row+1,col]
            left = None if col-1<0 else data_d[row,col-1]
            right = None if col+1>data_d.shape[1] else data_d[row,col+1]

            if data_d[row-1][col] == data_d[row+1][col]:
                fill_value =  data_d[row+1][col]
            elif data_d[row][col-1] == data_d[row][col+1]:
                fill_value =  data_d[row][col+1]
            else:
                scale_up_row = (data_d[row-1][col-1]+data_d[row-1][col+1])/data_d[row-1][col]
                scale_down_row = (data_d[row +1][col - 1] + data_d[row +1][col + 1]) / data_d[row + 1][col]

                scale =  (scale_up_row+ scale_down_row)/2
                fill_value = (data_d[row][col-1]+data_d[row][col+1])/scale

            data[item][row, col] = fill_value

    data.close()

if __name__ == '__main__':
    import argparse

    praser = argparse.ArgumentParser(prog='PROG')
    praser.add_argument("lut",type=str,help="lut name, aerosol,rayleigh,gas,path")
    praser.add_argument("--im", type=int, help="atmosphere type，1:'tropical',2:'midlatsummer',3:'midlatwinter',4:'subarcsummer',5:'subarcwinter'")
    praser.add_argument('--ia',type=int,help='aerosol type index')

    args = praser.parse_args()
    lut_name = args.lut

    # mode_name = {1:'tropical',2:'midlatsummer',3:'midlatwinter',4:'subarcsummer',5:'subarcwinter'}[args.im]
    aero_type = args.ia

    # fillNA(lut_name,aero_type,mode_name)
    fillNA_path(lut_name,aero_type)
