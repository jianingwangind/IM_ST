set -ex
python ./test.py --dataroot /data --name cifar2gta --model cycle_gan --phase test --no_dropout --dataset_mode unaligned --resize_or_crop crop --loadSize 400 --fineSize 32  --how_many 60000
