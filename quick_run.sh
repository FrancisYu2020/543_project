EXP=grab
CONFIG=celebatest
DATA_DIR=data/grab
OUPUT_DIR=data/grab_out
IMAGE_SIZE=128
CKPT=GAN2Shape/checkpoints/stylegan2/stylegan2-celeba-config-e.pt
STEP=10
CHANNEL_MULTIPLIER=1


python GAN2Shape/projector.py --size $IMAGE_SIZE --ckpt $CKPT -cm $CHANNEL_MULTIPLIER -out $OUPUT_DIR --step $STEP ${DATA_DIR}/*
python process_latent.py -in ${OUPUT_DIR}
cd GAN2Shape
GPUS=1
PORT=${PORT:-29579}

mkdir -p results/${EXP}
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    run.py \
    --launcher pytorch \
    --config configs/${CONFIG}.yml \
    2>&1 | tee results/${EXP}/log.txt
