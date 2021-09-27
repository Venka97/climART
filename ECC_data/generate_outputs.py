import argparse
import glob
import logging
import os
import sys
from multiprocessing import Pool
import shutil
import subprocess

import netCDF4
from tqdm import tqdm

# ncldy=ncldy*0; aerin=aerin*0.0f; sw_ext_sa=sw_ext_sa*0.0f; sw_ssa_sa=sw_ssa_sa*0.0f;
# sw_g_sa=sw_g_sa*0.0f; lw_abs_sa=lw_abs_sa*0.0f; ccld=ccld*0.0f; cldtrol=cldtrol*0.0f;
# clw_sub=clw_sub*0.0f; cic_sub=cic_sub*0.0f; rel_sub=rel_sub*0.0f;rei_sub=rei_sub*0.0f'
condition_dict = {
    'pristine': [
        'ncldy', 'aerin', 'sw_ext_sa', 'sw_ssa_sa',
        'sw_g_sa', 'lw_abs_sa', 'ccld', 'cldtrol',
        'clw_sub', 'cic_sub', 'rel_sub', 'rei_sub'
    ],
    'clear_sky': [
        'ncldy', 'ccld', 'cldtrol',
        'clw_sub', 'cic_sub', 'rel_sub', 'rei_sub'
    ],
    'all_sky': []
}
TMP_DIR = "/miniscratch/salva.ruhling-cachay/ECC_data/snapshots/1979-2014"
TMP_DIR = "/miniscratch/venkatesh.ramesh/ECC_data/"

def data_to_exp_type_for_rte_solver(input_filename: str, exp_type: str) -> str:
    condition = condition_dict[exp_type]
    if len(condition) == 0:
        return input_filename
    else:
        TMP_EXPTYPE_INPUT = f"{TMP_DIR.replace('/input_raw', '')}/tmp_{exp_type}_{os.getpid()}pid.nc"

        shutil.copyfile(input_filename, TMP_EXPTYPE_INPUT)  # DO NOT TOUCH ORIGINAL INPUT FILE
        data_nc = netCDF4.Dataset(TMP_EXPTYPE_INPUT, 'r+')  # 'w' will write/create a blank nc dataset in data_nc
        for col in condition:
            data_nc.variables[col][:] = 0
        data_nc.close()
        return TMP_EXPTYPE_INPUT.strip()


def create_config_file(input_filename: str, output_path: str) -> str:
    output_filename = output_path.split('/')[-1]
    output_dir = output_path.replace(output_filename, '').rstrip('/')
    os.makedirs(output_dir, exist_ok=True)
    cfg_filename = '/home/mila/v/venkatesh.ramesh/canam/single_column_models/rad_tran/cfg/rt_full_global_gcm.cfg'

    TMP_EXPTYPE_CFG = f"{TMP_DIR}/tmp_config_{os.getpid()}pid.run_info"
    with open(TMP_EXPTYPE_CFG, 'w') as conf:
        # conf.write("#!/bin/bash\n")
        conf.write("&runinfo\n")
        conf.write("cfg_filename    = " + f'"{cfg_filename}"' + "\n")
        conf.write("input_filename  = " + f'"{input_filename}"' + "\n")
        conf.write("output_dir = " + f'"{output_dir}"' + "\n")
        conf.write("output_filename = " + f'"{output_filename}"' + "\n")
        conf.write("/" + "\n")
        conf.write("\n")
    return TMP_EXPTYPE_CFG


def create_output(input_filename: str, exp_type: str, prefix='') -> None:
    output_path = input_filename.replace(f"{prefix}input_raw", f"{prefix}output_{exp_type.lower()}")

    # If output file already exists and is non-empty, there's no need to recreate it
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
        print(f"{output_path} exists already... skipping it.")
        return
    else:
        print(f"Generating {output_path}...")
    exptype_input_filename = data_to_exp_type_for_rte_solver(input_filename, exp_type=exp_type)

    config_fname = create_config_file(exptype_input_filename, output_path=output_path)

    bashCommand = f"/home/mila/v/venkatesh.ramesh/canam/single_column_models/rad_tran/build/scm_rt {config_fname}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if output:
        print(f"{os.getpid()}PID:\n{output}")
    if error:
        print(f"{os.getpid()}PID:\n{error}")
    sys.stdout.flush()

    return 1

