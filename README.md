# ***climART*** - Official Pytorch Implementation

### Using deep learning to optimise radiative transfer calculations.

Abstract:   *Numerical simulations of Earth's weather and climate require substantial amounts of computation. This has led to a growing interest in replacing subroutines that explicitly compute physical processes with approximate machine learning (ML) methods that are fast at inference time. Within weather and climate models, atmospheric radiative transfer (RT) calculations are especially expensive.  This has made them a popular target for neural network-based emulators. However, prior work is hard to compare due to the lack of a comprehensive dataset and standardized best practices for ML benchmarking. To fill this gap, we build a large dataset, ClimART, with more than **10 million** samples from present, pre-industrial, and future climate conditions}, based on the Canadian Earth System Model.
ClimART poses several methodological challenges for the ML community, such as multiple out-of-distribution test sets, underlying domain physics, and a trade-off between accuracy and inference speed. We also present several novel baselines that indicate shortcomings of datasets and network architectures used in prior work.*

Contact: Venkatesh Ramesh [venka97 at gmail] (mailto:venka97@gmail.com) or Salva Rühling Cachay [salvaruehling at gmail] (mailto:salvaruehling@gmail.com) <br>

## Directory Structure

```
.
├── analysis
├── ECC_data
├── notebooks
├── rtml
│   ├── data_wrangling
│   ├── models
│   ├── utils
├── tests
└── train_scripts
  
```

## Overview:

* ``rtml/``: Folder with the main code for RTML processing.
* ``notebooks/``: Notebooks for visualization of data.
* ``analysis/``: Scripts to create visualization of the results.
* ``scripts_to_run/``: Scripts to train and evaluate models on the dataset.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* NVIDIA GPUs with at least 8 GB of memory and system with 12 GB RAM (More RAM is required if training with ``--load_train_into_mem`` option which allows for faster training). We have done all testing and development using NVIDIA V100 GPUs.
* 64-bit Python 3.7 and PyTorch 1.8.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.  
* Python libraries mentioned in ``environment.yml`` file.

## Getting Started

Need to have miniconda/conda installed. 

1. Install: ``conda env create -f env.yml``
2. ``conda activate rtml_nips``
3. ``bash data_download.sh``
4. Go to the train_scripts folder.
5. Select the appropriate model script and run using bash ``script_name.sh``

## Tweaking the dataset

It's possible to change the dataset use for training the data. The steps are as follows:
1. Download the train/val/test years you want to use by changing the loop in ``data_download.sh.``
2. Go to ``rtml/data_wrangling/constants.py`` and change the constants to indicate your choice of years (between ``line 30. & 40.``)
3. Alternatively, traing and val data can be changed from the appropriate modelscript in ``train_scripts/`` folder.
4. Run with the new modified dataset.

## Training Options

```
--expID: A unique ID for the experiment if using logging.
--model: Model architecture to select for training (MLP, GCN, GN, CNN)
--scheduler: The learning rate scheduler used for training (expdecay, reducelronplateau, steplr, cosine)
--lr: The learning rate to use for training.
--optim: The choice of optimizer to use.
--weight_decay: Weight decay to use for Adam optimizer.
--batch_size: Batch size for training.
--dropout: Dropout to use in final layers of model.
--act: Activation of choice for trainng models.
--loss: Loss function to train the model with.
--epochs: Number of epochs to train the model for.
--in_normalize: Select how to normalize the data (Z, min_max, None)
--train_years: The years to select for training the data. (Either individual years 1997+1991 or range 1991-1996)
--validation_years: The years to select for validating the data. (Either individual years 1997+1991 or range 1991-1996)
--net_norm: Normalization scheme to use in the model (batch_norm, layer_norm, inst_norm)
--gap: Use global average pooling in-place of MLP to get output (CNN only).
--gradient_clipping: Value to clip the gradient to while training.
--hidden_dim: The hidden dimension to use for model.
--workers: The number of workers to use for dataloading.

```

## Baseline Configurations

### CNN

```
python ../main.py --expID "" --model CNN --scheduler "expdecay" --lr 2e-4 --weight_decay 1e-6 --batch_size 128 \
  --dropout 0.0 --act GELU --optim Adam --loss mse --epochs 100 --workers 8 --in_normalize Z \
  --train_years "1990+1999+2003" --validation_years "2005" --net_norm none \
  --gap --gradient_clipping norm --seed 7 \
  --wandb_mode disabled \
```

### MLP 

```
python ../main.py --expID "" --model "MLP" --scheduler "expdecay" --exp_type "pristine" --target_type "shortwave" --lr 2e-4 \
  --weight_decay 1e-6 --batch_size 128 --net_normalization "layer_norm" --dropout 0.0 --act "GELU" \
  --optim Adam --gradient_clipping "Norm" --clip 1 --epochs 100 --workers 6 --hidden_dims 512 256 256 \
  --in_normalize Z --train_years "1990+1999+2003" --validation_years "2005" --seed 7 \
  --wandb_mode disabled \
```

### GCN

```
python ../main.py \
  --model "GCN+Readout" --target_type "shortwave" --workers 8 --expID "" --hidden_dims 128 128 128 --scheduler "expdecay" \
  --lr 2e-4 --weight_decay 1e-6 --batch_size 128 --act "GELU" --net_normalization "none" --gradient_clipping "Norm" --clip 1.0 \
  --epochs 100 --residual --improved_self_loops --preprocessing "mlp_projection" --projector_net_normalization "layer_norm" \
  --graph_pooling "mean" --drop_last_level --in_normalize Z --train_years "1990+1999+2003" --validation_years "2005" \
  --wandb_mode disabled \
```

## Logging

Currently, logging is disabled by default. However, the user may use wandb to log the experiments by changing the wandb confing in main.py.

## Notebooks

There are some jupyter notebooks in the notebooks folder which we used for plotting, benchmarking etc. You may go through them to visualize the results/benchmark the models.

## License

This work is made available under [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license. 

## Development

This repository is currently under active development and you may encounter bugs with some functionality. Pull requests relating to bugs will be accepted however, we don't accept outside code contributions in terms of new features as this repository is a replication of the code related with the paper. 
