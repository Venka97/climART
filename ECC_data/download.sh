#!/bin/bash
#SBATCH --job-name=RT_Download
#SBATCH --output=job_output_RT_Download.txt
#SBATCH --error=job_error_RT_Download.txt
#SBATCH --ntasks=1
#SBATCH --time=25:00:00
#SBATCH --mem=64Gb
#SBATCH --partition=long
#SBATCH -c 8

download_dir="/miniscratch/salva.ruhling-cachay/ECC_data/snapshots/1979-2014/"
ftp_site="ftp://anonymous:pass@crd-data-donnees-rdc.ec.gc.ca///pub/CCCMA/jcole/machine_learning_rt"

declare -a periods=("1979-1984" "1985-1989" "1990-1994" "1995-1999" "2000-2004" "2005-2009" "2010-2014")

# Download/wget raw input data
for period in "${periods[@]}"; do
  echo Downloading ${ftp_site}/"${period}"_205hours
  wget -r -nH --cut-dirs=5 --directory-prefix=$download_dir -nc ${ftp_site}/"${period}"_205hours
done

# Unzip
shopt -s nullglob
for dir in "${download_dir}"machine_learning_rt/*/
do
    for file in "$dir"/*.nc.bz2
    do
        if [[ -f $file ]]
        then
            bzip2 -d "$file"
        fi
    done
done

module load anaconda/3
conda activate rtml-gnn
# Create pristine inputs and corresponding RTE outputs
python generate_outputs.py --path "${download_dir}" --condition pristine
