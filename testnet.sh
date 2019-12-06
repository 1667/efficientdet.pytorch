#!/bin/sh

# python testnet.py --model_file weights/VOC_efficientnet-b0_300_BiFPN_200.pth --backbone efficientnet-b0
python testnet.py --model_file weights/VOC_efficientnet-b2_half_300_BiFPN_100.pth --backbone efficientnet-b2 --half True
