#!/usr/bin/bash
python train_txt2im.py --mode 'train'

python train_txt2im.py --mode 'train_encoder'

python train_encoder --mode 'translation'
