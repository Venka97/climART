import logging

# Training settings
from functools import partial

import wandb
import torch
from torch.utils.data import DataLoader

from rtml.data_wrangling.constants import TEST_YEARS, LAYERS, get_flux_mean, OOD_PRESENT_YEARS, TRAIN_YEARS, \
    GLOBALS, LEVELS, PRISTINE, get_data_dims
from rtml.data_wrangling.h5_dataset import RT_HdF5_Dataset
from rtml.models.column_handler import ColumnPreprocesser
from rtml.models.interface import get_trainer, is_gnn, is_graph_net, get_model, get_input_transform

from rtml.utils.hyperparams_and_args import get_argparser
from rtml.utils.preprocessing import Normalizer
from rtml.utils.utils import set_seed, year_string_to_list, get_logger, get_target_variable, get_target_types

torch.set_printoptions(sci_mode=False)
log = get_logger(__name__)


def main(params, net_params, other_args, only_final_eval=False, *args, **kwargs):
    set_seed(params['seed'])

    # wandb.login()
    project = "RT+ML_Dataset_paper"
    # if 'longwave' in params['target_type']:
    #    project += '_LongWave'
    run = wandb.init(
        project=project,
        settings=wandb.Settings(start_method='fork'),
        tags=['random-train-batches'],
        entity="ecc-mila7",
        name=params['wandb_name'],
        group=params['ID'], reinit=True, mode=other_args.wandb_mode,
        id=params['wandb_ID'], resume="allow"
    )

    spatial_dim, in_dim = get_data_dims(params['exp_type'])

    if is_gnn(params['model']) or is_graph_net(params['model']):
        cp = ColumnPreprocesser(
            n_layers=spatial_dim[LAYERS], input_dims=in_dim, **params['preprocessing_dict']
        )
        input_transform = cp.get_preprocesser
    else:
        cp = None
        input_transform = partial(get_input_transform, model_class=get_model(params['model'], only_class=True))

    dataset_kwargs = dict(
        exp_type=params['exp_type'],
        target_type=params['target_type'],
        target_variable=params['target_variable'],
        input_transform=input_transform,
        input_normalization=params['in_normalize'],
        spatial_normalization_in=params['spatial_normalization_in'],
        log_scaling=params['log_scaling'],
    )
    train_years = year_string_to_list(params['train_years'])
    assert all([y in TRAIN_YEARS for y in train_years])
    train_set = RT_HdF5_Dataset(years=train_years, name='Train',
                                output_normalization=params['out_normalize'],
                                spatial_normalization_out=params['spatial_normalization_out'],
                                load_h5_into_mem=params['load_train_into_mem'],
                                **dataset_kwargs)
    val_set = RT_HdF5_Dataset(years=year_string_to_list(params['validation_years']), name='Val',
                              output_normalization=None,
                              load_h5_into_mem=params['load_val_into_mem'],
                              **dataset_kwargs)

    test_names = [f'Test_{test_year}' for test_year in TEST_YEARS]
    test_sets = [
        RT_HdF5_Dataset(years=[test_year], name=test_name, output_normalization=None, **dataset_kwargs)
        for test_year, test_name in zip(TEST_YEARS, test_names)
    ]
    ood_test_set = RT_HdF5_Dataset(years=OOD_PRESENT_YEARS, name='OOD Test',
                                   output_normalization=None,
                                   **dataset_kwargs)

    net_params['input_dim'] = train_set.input_dim
    net_params['spatial_dim'] = train_set.spatial_dim
    net_params['out_dim'] = train_set.output_dim
    params['target_type'] = get_target_types(params.pop('target_type'))
    log.info(f" {'Targets are' if len(params['target_type']) > 1 else 'Target is'} {' '.join(params['target_type'])}")
    params['target_variable'] = get_target_variable(params.pop('target_variable'))
    params['training_set_size'] = len(train_set)
    output_normalizer = train_set.output_normalizer
    output_postprocesser = train_set.output_variable_splitter
    # if params['train_on_raw_targets']:
    #    train_set.set_normalizer(data_type=OUTPUT, new_normalizer=None)
    if not isinstance(output_normalizer, Normalizer):
        log.info('Initializing out layer bias to output train dataset mean!')
        params['output_bias_mean_init'] = True
        # out_layer_bias = np.array([267.87377 for _ in range(50)] + [53.5887 for _ in range(50)])
        out_layer_bias = get_flux_mean()
    else:
        params['output_bias_mean_init'] = False
        out_layer_bias = None

    trainer_kwargs = dict(
        model_name=params['model'], model_params=net_params,
        device=params['device'], seed=params['seed'],
        model_dir=params['model_dir'],
        out_layer_bias=out_layer_bias,
        output_postprocesser=output_postprocesser,
        output_normalizer=output_normalizer,
    )
    if cp is not None:
        trainer_kwargs['column_preprocesser'] = cp

    print(net_params)
    trainer = get_trainer(**trainer_kwargs)

    dataloader_kwargs = {'pin_memory': True, 'num_workers': params['workers']}
    eval_batch_size = 512
    trainloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, **dataloader_kwargs)
    valloader = DataLoader(val_set, batch_size=eval_batch_size, **dataloader_kwargs)

    testloaders = [
        DataLoader(test_set, batch_size=eval_batch_size, **dataloader_kwargs) for test_set in test_sets
    ]
    ood_testloader = DataLoader(ood_test_set, batch_size=eval_batch_size, **dataloader_kwargs)


    # testloaders = [testloader, ood_testloader]
    # testloader_names = ['Test', 'OOD test']
    wandb.config.update({**net_params, **params})  # **train_set.vars_used_or_not},
    if not only_final_eval:
        best_valid = trainer.fit(trainloader, valloader,
                                 hyper_params=params,
                                 testloader=testloaders,
                                 testloader_names=test_names,
                                 ood_testloader=ood_testloader,
                                 *args, **kwargs)
        wandb.log({'Final/Best_Val_MAE': best_valid})
        log.info(f" Testing the best model as measured by validation performance (best={best_valid:.3f})")

    train_set.close()
    del train_set, trainloader, valloader

    if other_args.save_model_to_wandb in [True, 'true', 'True'] and not only_final_eval:
        wandb.save(trainer.save_model_filepath)

    final_test_kwargs = dict(use_best_model=True, verbose=True, model_verbose=False)
    if only_final_eval:
        final_test_kwargs = {**kwargs, **final_test_kwargs}

    final_test_stats = trainer.test(
        testloaders=testloaders,
        testloader_names=[f'Final_yearly/{name}' for name in test_names],
        aggregated_test_name='Final/Test', **final_test_kwargs
    )
    wandb.log(final_test_stats)
    final_ood_stats = trainer.test(
        testloaders=[ood_testloader],
        testloader_names=["Final/Test_OOD"], **final_test_kwargs
    )
    wandb.log(final_ood_stats)
    # for k, val in final_ood_stats.items():
    #    if isinstance(val, float):
    #        wandb.run.summary[k] = val

    run.finish()
    for dset in [val_set, ood_test_set] + test_sets:
        dset.close()

if __name__ == '__main__':
    logging.basicConfig()
    params, net_params, other_args = get_argparser()

    if other_args.resume_training_file is None and other_args.resume_ID is None:
        main(
            params, net_params, other_args
        )
    else:
        print('--------------------> Resuming training of', other_args.resume_training_file)
        saved_model = torch.load(other_args.resume_training_file)

        params_resume = saved_model['hyper_params']
        net_params_resume = saved_model['model_params']

        params_resume['epochs'] += other_args.additional_epochs
        for k, v in params.items():
            if k not in params_resume:
                params_resume[k] = v
        main(
            params_resume, net_params_resume, other_args,
            checkpoint=other_args.resume_training_file
        )
