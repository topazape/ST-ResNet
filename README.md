# ST-ResNet
ST-ResNet ([Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/abs/1610.00081)) implementaion with Pytorch, improved preprocessing speed over the original code.

![ST-ResNet](assets/images/1-s2.0-S0004370218300973-gr003_lrg.jpg)


## Acknowledgement
Most of the code in the model definition is based on [https://github.com/KL4805/STResNet-PyTorch](https://github.com/KL4805/STResNet-PyTorch), which is another ST-ResNet Pytorch implimentaion.


## Data
Copy *TaxiBJ* dataset under `TaxiBJ` dir.

> **Note**
> TaxiBJ dataset is currently unavailable due to circumstances at the data provider.
>
> See: [TaxiBJ21: An open crowd flow dataset based on Beijing taxi GPS trajectories](https://doi.org/10.1002/itl2.297)


## Example
```sh
python run.py [-h] [-s SEED] FILE

positional arguments:
  FILE                  path to config file

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  seed for initializing training
```

## Config
The following is a setting for L4-C3-P1-T1, means *four* residual blocks, *three* closeness time steps, one period time step, one trend time step.

```ini
[dataset]
T = 48
len_closeness = 3
len_period = 1
len_trend = 1
period_interval = 1
trend_interval = 7
; 48 * 28
len_test = 1344
use_meta = true
use_holiday = true
use_meteorol = true

[model]
nb_flow = 2
map_height = 32
map_width = 32
nb_residual_unit = 12

[learning]
epochs = 100
batch_size = 32
learning_rate = 0.0002
```

## Reference
J. Zhang, Y. Zheng and D. Qi, "Deep spatio-temporal residual networks for citywide crowd flows prediction", AAAI, pp. 1655-1661, 2017.
