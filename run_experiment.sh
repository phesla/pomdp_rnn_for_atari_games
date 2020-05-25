#!/bin/bash

PYTHONPATH=. python3 ./cli/test.py --config_path=./configs/self_attn_cnn_gru.yml
PYTHONPATH=. python3 ./cli/train.py --config_path=./configs/self_attn_cnn_gru.yml
PYTHONPATH=. python3 ./cli/test.py --config_path=./configs/self_attn_cnn_gru.yml