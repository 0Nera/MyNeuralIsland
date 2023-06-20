#!/bin/sh
python3 make_dataset.py
python3 train.py
python3 upscale.py 