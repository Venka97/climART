import sys
import glob
import os

import xarray as xr
import numpy as np
import einops
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio

'''
    Command to run the code: python generate_data_vis.py 2003
'''

coords_data = xr.open_dataset('/miniscratch/venkatesh.ramesh/ECC_data/snapshots/coords_data/areacella_fx_CanESM5_amip_r1i1p1f1_gn.nc')

def generate_grid(file, save_path):
    #Get the TOA data from the file
    data_out = xr.open_dataset(file, drop_variables = 'iseed')
    data_out = data_out.stack(site=("latitude", "longitude"))
    data_out = data_out.transpose('site', ...)
    rsdc = data_out['rsdc']
    rsdc_lev = einops.rearrange(np.array(rsdc), 'p l -> l p')
    toa = rsdc_lev[0]
    toa = toa.reshape(64, -1)

    #Get lat-lon grid
    lat = list(coords_data.get_index('lat'))
    lon = list(coords_data.get_index('lon'))
    lon_g, lat_g = np.meshgrid(lon, lat)

    assert lon_g.shape == lat_g.shape == toa.shape
    grid_plot(toa, lon_g, lat_g, file, save_path)


def grid_plot(data, lon_grid, lat_grid, file, save_path):
    norm = TwoSlopeNorm(200, vmin=0, vmax=1000)

    fig = plt.figure(figsize=(12,8))
    # plt.rcParams['figure.dpi'] = 200

    ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
    ax.add_feature(cfeature.COASTLINE)
    ax.stock_img()
    ax.set_global()

    plt.pcolormesh(lon_grid, lat_grid, data, cmap='RdBu_r', norm=TwoSlopeNorm(100, vmin=-50, vmax=1200),
                transform=ccrs.PlateCarree())
    # print(os.path.join(save_path, file.split('/')[-1].replace('.nc', '.png')))
    plt.savefig(os.path.join(save_path, file.split('/')[-1].replace('.nc', '.png')), dpi=200)
    plt.close()

def create_gif(save_path):
    files = glob.glob(save_path + '/**/*.png', recursive=True)
    files.sort()
    print(f"Saving the GIF...")
    with imageio.get_writer(os.path.join(save_path, '2003.gif'), mode='I') as writer:
        for f in files:
            image = imageio.imread(f)
            writer.append_data(image)


if __name__=='__main__':
    year = sys.argv[1]

    if year == '2003':
        data_path = '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/output_pristine/'
        save_path = '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/visualization/2003'
    elif year == '2004':
        data_path = '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/2004/output_pristine/'
        save_path = '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/visualization/2004'
    else:
        data_path = '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/2005/output_pristine/'
        save_path = '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/visualization/2005'

    input_files = glob.glob(data_path + '/**/*.nc', recursive=True)
    print(f'Generating plots for {len(input_files)} files...')
    for f in tqdm(input_files):
        generate_grid(f, save_path=save_path)
    create_gif(save_path)
