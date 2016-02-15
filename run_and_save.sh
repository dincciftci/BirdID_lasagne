#!/bin/bash
rm *.pyc
sudo nohup python -u train_net.py -c sx3_ffc -s state_ffc > ffc_output.txt
sudo cp ffc_output.txt state_ffc.npy /data
sudo nohup python -u train_net.py -c sx3_ffc_b5 -s state_ffc_b5 > ffc_b5_output.txt
sudo cp ffc_b5_output.txt state_ffc_b5.npy /data
sudo nohup python -u train_net.py -c sx3_fffc -s state_fffc > fffc_output.txt
sudo cp fffc_output.txt state_fffc.npy /data
sudo shutdown -h now
