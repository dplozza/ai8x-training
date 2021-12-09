python train_test.py ^
    --epochs 2 --deterministic --compress schedule.yaml ^
    --model ai85testwavenet --dataset PEDALNET ^
    --device MAX78000 ^
    --regression ^
    --custom-loss ^
    --num-hidden-channels 8 ^
    --dilation-depth 10 ^
    --dilation-power 2 ^
    --dither_std 0.5 ^
    --dither_hicutoff 14000 ^
    --name template_testwavenet

Rem python train.py --epochs 1 --optimizer Adam --lr 0.001 --deterministic --compress schedule.yaml --model ai85kws20net --dataset KWS_20 --confusion --device MAX78000
