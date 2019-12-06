#!/bin/sh

# python train_fire.py --resume weights/VOC_efficientnet-b0_300_BiFPN_200.pth --num_epochs 200 --batch_size 48 --backbone efficientnet-b0
python train_fire.py --resume weights/VOC_efficientnet-b2_half_300_BiFPN_100.pth --num_epochs 200 --batch_size 64 --backbone efficientnet-b2 --half True
