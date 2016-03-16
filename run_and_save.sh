#!/bin/bash
rm *.pyc
sudo nohup python -u optimize.py -c sx3_ffc_b32 > ffc_optimize.txt
sudo cp ffc_optimize.txt /data
sudo shutdown -h now
