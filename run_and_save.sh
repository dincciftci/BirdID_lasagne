#!/bin/bash
nohup python -u train_net.py -c sx3_ffc -s state_ffc > ffc_output.txt
scp ffc_output.txt state_ffc.npy dc9wp@141.166.207.143:~/aws_output/
shutdown
