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
The following is a setting for L4-C3-P1-T1, means *four* residual blocks, *three* closeness time steps, *one* period time step and *one* trend time step.

```ini
[dataset]
T = 48			; time steps of the day. T=48 means 24 * 60 / 48 = 30 min = one time step
len_closeness = 3	; number of time steps used as closeness
len_period = 1		; number of time steps used as period
len_trend = 1		; number of time steps used as trend
period_interval = 1	; 1 specifies 1 day interval: 1 * T = 1 * 48 * 30 min = 24 hr = 1 day
trend_interval = 7	; 7 specifies 1 week interval: 7 * T = 7 * 48 * 30 min = 7 days = 1 wk
len_test = 1344		; number of test data
use_meta = true		; whether to use day of the week and weekend information
use_holiday = true	; use holiday information
use_meteorol = true	; use weather information

[model]
nb_flow = 2		; number of channels: 2 means the number of in/out-flows
map_height = 32		; grid height
map_width = 32		; grid width
nb_residual_unit = 12	; number of residual blocks

[learning]
epochs = 100
batch_size = 32
learning_rate = 0.0002
```

## Reference
J. Zhang, Y. Zheng and D. Qi, "Deep spatio-temporal residual networks for citywide crowd flows prediction", AAAI, pp. 1655-1661, 2017.
