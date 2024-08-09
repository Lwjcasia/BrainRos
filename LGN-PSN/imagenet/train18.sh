# train imagenet script 
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m torch.distributed.launch --nproc_per_node=4 --master_port 1266 --use_env train.py --cos_lr_T 320 --model sew_resnet18 \
-b 128 --output-dir ./logs --tb --print-freq 1024 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 \
--data-path /mnt/data/dfxue/datasets/ILSVRC/Data/CLS-LOC --tet -j 4 \
--load /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/imagenet/pretrained/resnet18-f37072fd.pth \
# --resume /mnt/data/dfxue/disk/codes-snn/Parallel-Spiking-Neuron/imagenet/logs/sew_resnet18_b128_lr0.1_T4_coslr320_sgd_4gpu_load_tet/checkpoing_last.pth
