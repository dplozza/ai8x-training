python train_test.py ^
    --epochs 1 --deterministic --compress schedule-nothing.yaml ^
    --model ai85reswavenet --dataset PEDALNET ^
    --device MAX78000 ^
    --regression ^
    --custom-loss ^
    --num-hidden-channels 2

Rem python train.py --epochs 1 --optimizer Adam --lr 0.001 --deterministic --compress schedule.yaml --model ai85kws20net --dataset KWS_20 --confusion --device MAX78000
