#!/bin/bash

#ConvAttention model, Partial data
python Conv2D.py --include_type SC,ST --data_ratio 0.25 --model 1

#ConvAttention3 model, Partial data
python Conv2D.py --include_type SC,ST --data_ratio 0.25 --model 0 --num_epoch 20



#ConvAttention model, Partial data
python Conv2D.py --include_type SC,ST --data_ratio 1 --model 1

#ConvAttention3 model, Partial data
python Conv2D.py --include_type SC,ST --data_ratio 1 --model 0 