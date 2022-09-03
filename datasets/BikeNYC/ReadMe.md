BikeNYC: New-Flow/End-Flow at NYC
===========================================================

**If you use the data, please cite the following paper.**

`Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017. `

Download data from [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhCtwaiDRy5oDVIug) or [BaiduYun](http://pan.baidu.com/s/1mhIPrRE)

Please check the data with `md5sum` command: 
```
md5sum -c md5sum.txt
```

**BikeNYC** dataset is a `hdf5` file named `NYC14_M16x8_T60_NewEnd.h5`, which includes two subsets:

* `date`: a list of timeslots, which is associated the **data**. 
* `data`: a 4D tensor of shape (number_of_timeslots, 2, 16, 8), of which `data[i]` is a 3D tensor of shape (2, 16, 8) at the timeslot `date[i]`, `data[i][0]` is a `16x8` new-flow matrix and `data[i][1]` is a `16x8` end-flow matrix. 

Note: `*.h5` is `hdf5` file, one can use the follow code to view the data:

```
import h5py
f = h5py.File('NYC14_M16x8_T60_NewEnd.h5')
for ke in f.keys():
    print(ke, f[ke].shape)
```
