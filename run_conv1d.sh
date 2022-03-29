#!/bin/bash

#ConvAttention model, Partial data
#python Conv1D.py --include_type SC,ST --data_ratio 0.25 --model 1

#ConvASPP model, Partial data
#python Conv1D.py --include_type SC,ST --data_ratio 0.25 --model 3 

#Resnet34 model, Partial data
#python Conv1D.py --include_type SC,ST --data_ratio 0.25 --model 0 


#ConvAttention model, Full data
#python Conv1D.py --include_type SC,ST --data_ratio 1.0 --model 1 

#ConvASPP model, Full data
#python Conv1D.py --include_type SC,ST --data_ratio 1.0 --model 3 

#Resnet34 model, Full data
#python Conv1D.py --include_type SC,ST --data_ratio 1.0 --model 0 


#ConvASPP model, SC data
#python Conv1D.py --include_type SC --data_ratio 1.0 --model 3

#ConvAttention model, SC data
#python Conv1D.py --include_type SC --data_ratio 1.0 --model 1

#ConvASPP_1(With only small field-of views) model, Partial data
#python Conv1D.py --include_type SC,ST --data_ratio 0.25 --model 4

#ConvASPP_1(With only small field-of views) model, Partial data
#python Conv1D.py --include_type SC,ST --data_ratio 1.0 --model 4

#ConvASPP_1(With only small field-of views) model, SC data
#python Conv1D.py --include_type SC --data_ratio 1.0 --model 4

#ConvASPP_1(With only small field-of views) model, SC data
#python Conv1D.py --include_type SC --data_ratio 1.0 --model 4

#Resnet34 model, SC data
#python Conv1D.py --include_type SC --data_ratio 1.0 --model 0

#ConvASPPAttention model, SC, ST data
#python Conv1D.py --include_type SC,ST --data_ratio 0.25 --model 5 

#ConvASPPAttention model, SC, ST data
#python Conv1D.py --include_type SC,ST --data_ratio 1.0 --model 5


#ConvASPPAttention model, SC, ST data
python Conv1D.py --include_type SC,ST --data_ratio 0.25 --model 7

#ConvASPPAttention model, SC, ST data
python Conv1D.py --include_type SC,ST --data_ratio 1.0 --model 7