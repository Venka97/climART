#TODO
with open('/Users/Venky/Downloads/Canam_data/files_1979_1984.txt') as f:
    files = f.readlines()
cfg_filename = '"/home/mila/v/venkatesh.ramesh/canam/single_column_models/rad_tran/cfg/rt_full_global_gcm.cfg"'
output_dir = '"//miniscratch/venkatesh.ramesh/ECC_data/snapshots/all_data/machine_learning_rt/pristine/1979_1984_205hours/output/"'
run_canam = []

for f in files:
    fname = f.strip()
    outname = fname.split('.')[0]
    input_filename = "/miniscratch/venkatesh.ramesh/ECC_data/snapshots/all_data/machine_learning_rt/pristine/1979_1984_205hours/input/{}".format(
        fname)
    output_filename = outname + "_output_.nc"
    with open(
            '/Users/Venky/Downloads/Canam_data/confs/pristine/rt_full_global_gcm' + "pristine_" + outname + '.run_info',
            'w') as conf:
        # conf.write('Hello\n')
        conf.write("#!/bin/bash\n")
        conf.write("&runinfo\n")
        conf.write("cfg_filename    = " + cfg_filename + "\n")
        conf.write("input_filename  = " + '"{}"'.format(input_filename) + "\n")
        conf.write("output_dir = " + output_dir + "\n")
        conf.write("output_filename = " + '"{}"'.format(output_filename) + "\n")
        conf.write("/" + "\n")
        conf.write("\n")
    run_canam.append(
        "build/scm_rt" + " " "cfg/pristine/{}".format('rt_full_global_gcmpristine_' + outname + '.run_info'))
with open('/Users/Venky/Downloads/Canam_data/confs/pristine/run_canam_pristine_all.txt', 'w') as f:
    for item in run_canam:
        f.write("%s\n" % item)
