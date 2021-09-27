#!/bin/bash
#SBATCH --job-name=ZIP
#SBATCH --output=ZIP_out.txt
#SBATCH --error=ZIP_error.txt
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=15Gb
#SBATCH --gres=gpu:1
#SBATCH --partition=unkillable
#SBATCH -c 2

data_dir="/miniscratch/salva.ruhling-cachay/ECC_data/snapshots/1979-2014/hdf5"

mv "${data_dir}"/inputs/1992.h5 "${data_dir}"/NOT_INCLUDE/inputs/1992.h5
mv "${data_dir}"/inputs/1993.h5 "${data_dir}"/NOT_INCLUDE/inputs/1993.h5

mv "${data_dir}"/outputs_pristine/1992.h5 "${data_dir}"/NOT_INCLUDE/outputs_pristine/1992.h5
mv "${data_dir}"/outputs_pristine/1993.h5 "${data_dir}"/NOT_INCLUDE/outputs_pristine/1993.h5

mv "${data_dir}"/outputs_clear_sky/1992.h5 "${data_dir}"/NOT_INCLUDE/outputs_clear_sky/1992.h5
mv "${data_dir}"/outputs_clear_sky/1993.h5 "${data_dir}"/NOT_INCLUDE/outputs_clear_sky/1993.h5

tar -zcvf climart_present_day_data.tar.gz \
  "${data_dir}"/statistics.npz  \
  "${data_dir}"/inputs  \
  "${data_dir}"/outputs_pristine \
  "${data_dir}"/outputs_clear_sky
