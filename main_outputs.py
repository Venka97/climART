import argparse
import glob
import logging
import os
import sys
from functools import partial
from multiprocessing import Pool

from ECC_data.generate_outputs import create_output

logging.basicConfig()
TMP_DIR = "/miniscratch/venkatesh.ramesh/ECC_data/historic/input_raw"

parser = argparse.ArgumentParser(description='NetCDF input conditions preparation')
parser.add_argument('--path', type=str, default=TMP_DIR, help='path where the input .nc files are stored')
parser.add_argument('--exp_type', type=str, default='pristine',
                    help='The type of atmospheric condition (one of: clear or pristine)')

args = parser.parse_args()
exp_type = args.exp_type.lower()

files = glob.glob(args.path + '/**/CanAM_snapshot*.nc', recursive=True)
print(f'There are {len(files)} files to be processed')
n_cpus = 8  # os.cpu_count()
print('#CPUs =', os.cpu_count())
if n_cpus > 0:
    pool = Pool(n_cpus)
    # pool.imap()
    results = pool.map(partial(create_output, exp_type=exp_type), files)
    pool.close()
    print(results)
else:
    for file in files:
        print('going for', file)
        create_output(file, exp_type=exp_type)
sys.stdout.flush()

