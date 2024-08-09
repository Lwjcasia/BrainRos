python train_cf10.py -data-dir /mnt/data/dfxue/datasets/cifar10 -amp -opt sgd -channels 256 -epochs 1024 -device cuda:5 -T 4 \
-resume /mnt/data/dfxue/disk/codes-snn/LGN-PSN/cifar10/logs_cf10/pt/T4_e1024_b128_sgd_lr0.1_c256_amp_lg2_lsbr8*_gmean*_afterpool_lossf1e-7/checkpoint_max.pth \
-test_only 

# base: /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/cifar10/logs_base/if5_T4_e1024_b128_sgd_lr0.1_c256_amp/
# lgn-s /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/cifar10/logs_cf10/pt/T4_e1024_b128_sgd_lr0.1_c256_amp_lg2_lsbr16\*_gmean\*_afterpool_lossf1e-6/
# lgn-l /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/cifar10/logs_cf10/pt/T4_e1024_b128_sgd_lr0.1_c256_amp_lg2_lsbr8\*_gmean\*_afterpool_lossf1e-7/
