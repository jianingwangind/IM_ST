set -ex
python ./train.py --dataroot /data --name cifar2gta5 --model cycle_gan --pool_size 50 --no_dropout --resize_or_crop crop --loadSize 400 --fineSize 32 --niter 20 --niter_decay 0
