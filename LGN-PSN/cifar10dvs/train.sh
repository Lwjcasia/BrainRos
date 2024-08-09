export CUDA_VISIBLE_DEVICES=6,7
# python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 4 #-resume /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/cifar10dvs/logs/T4_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu_l8_lsbr8*+_s50/checkpoint_latest.pth
# python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 8 #-resume /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/cifar10dvs/logs/T8_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu/checkpoint_latest.pth
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 10 -resume /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/cifar10dvs/logs/T10_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu_lg8_lsbr32\*_gmean\*_beforeneu/checkpoint_latest.pth
