# POMDP RNN for atari games.

## We have made simple pipeline to make and train reinforcement agent models to aproximate POMDP by convolution and recurrent neural networks.

## To run train
Just change a config file if you want it (`./configs/...`).
To run training process you shoud change paths to the local paths (see config what you will use) and then run:
```
PYTHONPATH=. python cli/train.py --config_path=./configs/your_config.yml
```

## To run test
You can write video by running test script after (for example) 10 iterations (videos will be saved by defined path). Then change your config path to the new saved agent model path and run training process again. Then run test script second time to compare agent quality.
To run testing process you shoud change paths to the local paths (see config what you will use) and then run:
```
PYTHONPATH=. python cli/test.py --config_path=./configs/your_config.yml
```
