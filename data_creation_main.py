import logging
from rtml.data_wrangling.data_loading1979to84 import create_yearly_h5_files
from rtml.data_wrangling.precompute_dataset_stats import pre_compute_dataset_statistics_on_h5
from rtml.utils.utils import get_logger

log = get_logger(__name__)

if __name__ == '__main__':
    logging.basicConfig()
    # data_dir = "/miniscratch/salva.ruhling-cachay/ECC_data/snapshots/1979-2014"

    future_dir = "/miniscratch/venkatesh.ramesh/ECC_data/future"
    cpus = 0
    first_val_year = 2005

    which_years = 'all'
    # which_years = [1995, 2001, 2002, 2004, 2006]
    which_years = [1850, 1851, 1852] #, 2097, 2098, 2099]
    # which_years = [2097, 2098, 2099]
    # which_years = [2099]
    which_years = [2099]
    CREATE_H5 = True
    COMPUTE_STATS = False
    if CREATE_H5:
        expID = create_yearly_h5_files(
            future_dir,  # data_dir
            val_year_start=first_val_year,
            test_year_start=2007,
            test_pinatubo=True,
            train_files_per_year='all',
            val_files_per_year='all',
            test_files_per_year='all', #15,
            which_years=which_years,
            multiprocessing_cpus=cpus
        )

        log.info('Done with H5 creation! Computing variable statistics now...')

    training_years = list(range(1979, 1991)) + list(range(1994, first_val_year))

    if COMPUTE_STATS:
        pre_compute_dataset_statistics_on_h5(
            training_years=training_years,
            compute_spatial_stats_too=True
        )