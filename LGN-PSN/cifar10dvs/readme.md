## Dependencies

The codes in this repo are modified from the PSN paper.

The codes in this repo require a specific modified SpikingJelly. Install this specific SpikingJelly similar to [PSN](https://github.com/fangwei123456/Parallel-Spiking-Neuron/tree/main/cifar10dvs).

## Usage

```
usage: train_vgg.py [-h] [-j N] [--epochs N] [--start_epoch N] [-b N] [--lr LR] [--seed SEED] [-T N] [--means N] [--lamb N]
                    [-out_dir OUT_DIR] [-resume RESUME] [-method METHOD] [-opt OPT] [-tau TAU] [-TET]

PyTorch Training

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 10)
  --epochs N            number of total epochs to run
  --start_epoch N       manual epoch number (useful on restarts)
  -b N, --batch_size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel
  --lr LR, --learning_rate LR
                        initial learning rate
  --seed SEED           seed for initializing training.
  -T N                  snn simulation time (default: 2)
  --means N             make all the potential increment around the means (default: 1.0)
  --lamb N              adjust the norm factor to avoid outlier (default: 0.0)
  -out_dir OUT_DIR      root dir for saving logs and checkpoint
  -resume RESUME        resume from the checkpoint path
  -method METHOD        use which network
  -opt OPT              optimizer method
  -tau TAU              tau of LIF
  -TET                  use the tet loss
```

The options used in the paper are

```
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 4
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 8
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 10
```

And the number of GPUs is 2.

The original terminal outputs are saved in

```
T4_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu.log
T8_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu.log
T10_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu.log
```

