EXP=celebatest
CONFIG=celebatest
DATA_DIR=data/celebatest
OUPUT_DIR=data/celebasub
IMAGE_SIZE=128
CKPT=GAN2Shape/checkpoints/stylegan2/stylegan2-celeba-config-e.pt
CHANNEL_MULTIPLIER=1


python GAN2Shape/projector --size $IMAGE_SIZE --ckpt $CKPT -cm $CHANNEL_MULTIPLIER -out $OUPUT_DIR $DATA_DIR
python process_latent.py -in $OUPUT_DIR
cd GAN2Shape
sh scripts/run_celebatest.sh

