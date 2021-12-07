# Training script starter

cd ..
cd ..

COMMENT="_test"

MODEL=ai85simplewavenet
MODEL_PREFIX=resimple #resskip res 
DATASET=PEDALNET
DATASET_PREFIX="" #else _nam

EPOCHS=110
BATCH_SIZE=128
LR="1e-3"
OPTIMIZER=ADAM

QAT_POLICY=qat_policy_wavenet_early.yaml
QAT_PREFIX="_qearly" #"_q"
SCHEDULE=schedule-wavenet-early-decay.yaml

CHANNELS=12
DIL_DEPTH=8
DIL_POWER=2
OUT_BITDEPTH=8

DITHER_STD=0
DITHER_HICUTOFF=0

NAME=${MODEL_PREFIX}${DATASET_PREFIX}_ch${CHANNELS}_d${DIL_DEPTH}p${DIL_POWER}_lr${LR}_dith${DITHER_STD}hi${DITHER_HICUTOFF}${QAT_PREFIX}${COMMENT}
python ./train_test.py --deterministic --device=MAX78000 --regression --custom-loss --use-bias --enable-tensorboard --model=$MODEL --dataset=$DATASET --epochs=$EPOCHS --batch-size=$BATCH_SIZE --optimizer=$OPTIMIZER --lr=$LR --qat-policy=$QAT_POLICY --compress=$COMPRESS --num-hidden-channels=$CHANNELS --dilation-depth=$DIL_DEPTH --dilation-power=$DIL_POWER --output-bitdepth=$OUT_BITDEPTH --dither-std=$DITHER_STD --dither-hicutoff=$DITHER_HICUTOFF --name=$NAME 


