# Important 

This repo is a **fork** of [alecwangcq/GraSP](https://github.com/alecwangcq/GraSP) that I modified to my needs. **The actual implementation is not mine**, but the original author's. The corresponding paper to the method can be found on ArXiV: [Wang et al. (2020)](https://arxiv.org/abs/2002.07376). 

# Changes I made

Since I only need this code to sanity-check my implementation of GraSP, I removed everything that was related to training or related to ImageNet. This means basically all the files except for Pruner/GraSp.

* Added functionality to work with "my" dataloaders, `timm` models and Python 3.10.8 / PyTorch 1.13.1.
* Tried to add some comments to make the code more understandable, since it is very hard to read imho
* Added layerwise print functionality, displaying the `numel()` and the `sum()` of the masks in layers that are relevant to pruning

# Strange behaviour 

Besides not being able to relate some parts of the code to the paper, I noticed strange behaviour with the pruning algorithm itself. It appears as if the assumed pruning metric does not actually lead to the correct amount of pruned values, until some threshold is met. Below the threshold, the algorithm prunes much fewer weights than it is supposed to. Since I haven't modified the inner workings of the scoring algorithm at all, this raises some questions I want to investigate. From a first investigation it appears as if the threshold only works if it is larger than 0. 

Below are three examplary pruning results for a pruning level of 50%, 97% and 98% respectively. I used the resnet10t model from `timm`, the CIFAR10 dataset with a batch size of 128. The results can be repiicated by simply running the main function. Note that for 50% and 97%, the results are identical, but they change dramatically for a pruning ratio of 98%.

## 50%

```
================================================================
| Layer            | Before        | After         | Ratio     |
================================================================
| -                | 648           | 324.0         | 0.5       |
| - conv1          | 6912          | 3502.0        | 0.5067    |
| - conv1.0        | 18432         | 9340.0        | 0.5067    |
| - conv1.1        | 36864         | 36864.0       | 1.0       |
| - conv1.2        | 36864         | 36864.0       | 1.0       |
| - conv1.3        | 73728         | 73728.0       | 1.0       |
| - conv1.4        | 147456        | 147456.0      | 1.0       |
| - conv1.5        | 8192          | 4159.0        | 0.5077    |
| - conv1.6        | 294912        | 294912.0      | 1.0       |
| - bn1            | 589824        | 589824.0      | 1.0       |
| - act1           | 32768         | 16379.0       | 0.4998    |
| - maxpool        | 1179648       | 1179648.0     | 1.0       |
| - layer1         | 2359296       | 2359296.0     | 1.0       |
| - layer1.0       | 131072        | 65750.0       | 0.5016    |
| - layer1.0.conv1 | 5120          | 2351.0        | 0.4592    |
================================================================

- Intended prune ratio: 0.5
- Actual prune ratio:   0.020590092601472376
- Threshold:            0.0
```

## 97% 

```
================================================================
| Layer            | Before        | After         | Ratio     |
================================================================
| -                | 648           | 323.0         | 0.4985    |
| - conv1          | 6912          | 3538.0        | 0.5119    |
| - conv1.0        | 18432         | 9252.0        | 0.502     |
| - conv1.1        | 36864         | 36864.0       | 1.0       |
| - conv1.2        | 36864         | 36864.0       | 1.0       |
| - conv1.3        | 73728         | 73728.0       | 1.0       |
| - conv1.4        | 147456        | 147456.0      | 1.0       |
| - conv1.5        | 8192          | 4079.0        | 0.4979    |
| - conv1.6        | 294912        | 294912.0      | 1.0       |
| - bn1            | 589824        | 589824.0      | 1.0       |
| - act1           | 32768         | 16471.0       | 0.5027    |
| - maxpool        | 1179648       | 1179648.0     | 1.0       |
| - layer1         | 2359296       | 2359296.0     | 1.0       |
| - layer1.0       | 131072        | 65737.0       | 0.5015    |
| - layer1.0.conv1 | 5120          | 2344.0        | 0.4578    |
================================================================

- Intended prune ratio: 0.97
- Actual prune ratio:   0.02060248660228825
- Threshold:            -0.0
```

## 98% 

```
================================================================
| Layer            | Before        | After         | Ratio     |
================================================================
| -                | 648           | 328.0         | 0.5062    |
| - conv1          | 6912          | 3485.0        | 0.5042    |
| - conv1.0        | 18432         | 9163.0        | 0.4971    |
| - conv1.1        | 36864         | 0.0           | 0.0       |
| - conv1.2        | 36864         | 0.0           | 0.0       |
| - conv1.3        | 73728         | 0.0           | 0.0       |
| - conv1.4        | 147456        | 0.0           | 0.0       |
| - conv1.5        | 8192          | 4123.0        | 0.5033    |
| - conv1.6        | 294912        | 0.0           | 0.0       |
| - bn1            | 589824        | 0.0           | 0.0       |
| - act1           | 32768         | 16159.0       | 0.4931    |
| - maxpool        | 1179648       | 0.0           | 0.0       |
| - layer1         | 2359296       | 0.0           | 0.0       |
| - layer1.0       | 131072        | 62906.0       | 0.4799    |
| - layer1.0.conv1 | 5120          | 2270.0        | 0.4434    |
================================================================

- Intended prune ratio: 0.98
- Actual prune ratio:   0.9800001462898457
- Threshold:            0.15410423278808594
```