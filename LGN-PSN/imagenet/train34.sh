# train imagenet script 
export CUDA_VISIBLE_DEVICES=4,5,6,7 
python -m torch.distributed.launch --nproc_per_node=4 --master_port 1269 --use_env train.py --cos_lr_T 320 --model sew_resnet34 \
-b 64 --output-dir ./logs --tb --print-freq 2048 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 \
--data-path /mnt/data/dfxue/datasets/ILSVRC/Data/CLS-LOC --tet -j 1 \
--load /mnt/data/dfxue/disk/codes-snn/LGN-PSN/imagenet/pretrained/resnet34-b627a593.pth \
--resume /mnt/data/dfxue/disk/codes-snn/code238_server/Parallel-Spiking-Neuron/imagenet/logs/sew_resnet34_b64_lr0.1_T4_coslr320_sgd_4gpu_load_tet_lsbr8\*_gmean\*/checkpoint_max_test_acc1.pth
