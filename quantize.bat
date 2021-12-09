call activate ai8x-synthesis
python ../ai8x-synthesis/quantize.py %1.pth.tar %1_q.pth.tar --device MAX78000 -v
call activate ai8x-training

