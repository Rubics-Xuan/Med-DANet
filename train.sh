

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 1831 \
train.py --seed 500 --gpu 0,1,2,3  --lr 0.0002 --end_epoch 350 --experiment BraTS2019
