# AGC Optimizers

A small lib for using adaptive gradient clipping in your optimizer.

- [News](#news)
- [Introduction](#introduction)
- [Comparison](#comparison)
- [To Do](#todo)

## News

### Sep 14, 2021
* Add AdamW, Adam, SGD and RMSprop with AGC 
* Add first comparsion between optimizers with and without AGC based on CIFAR10

## Introduction

[Brock et al.](https://arxiv.org/pdf/2102.06171.pdf) introduced 2021 a new clipping technique in order to increase stability of large batch training and high learning rates in their Normalizer-Free Networks (NFNet), the adaptive gradient clipping. This clipping method is not implemented in leading frameworks, thus I provide optimizers which are capable of AGC. 

### Installation
```bash
pip install agc_optims
```

### Usage

To be consistent with PyTorch all arguments of the optimizer remain the same as in the standard. Only two parameters are added for the AGC:
* `clipping` : Hyperparameter for the clipping of the parameter. Default value 1e-2, smaller batchsizes demand a higher clipping parameter
* `agc_eps` : Term used in AGC to prevent grads clipped to zero, default value 1e-3

#### SGD
```python
from agc_optims.optim import SGD_AGC

net = Net() # your model

optimizer = SGD_AGC(net.parameters(), lr=0.01, momentum=0.9, clipping=0.16)
```
#### Adam
```python
from agc_optims.optim import Adam_AGC

net = Net() # your model

optimizer = Adam_AGC(net.parameters(), lr=0.001, weight_decay=1e-4, clipping=0.16)
```
#### AdamW
```python
from agc_optims.optim import AdamW_AGC

net = Net() # your model

optimizer = AdamW_AGC(net.parameters(), lr=0.001, weight_decay=1e-4, clipping=0.16)
```
#### RMSprop
```python
from agc_optims.optim import RMSprop_AGC

net = Net() # your model

optimizer = RMSprop_AGC(net.parameters(), lr=0.001, clipping=0.16)
```
## Comparison

### To Do
- [x] Add first comparsion based on CIFAR10 with a small CNN
- [] Add comparsion based on CIFAR100 with a small CNN
- [] Add comparsion based on CIFAR10/100 with ResNet
- [] Add all optimizer included in PyTorch
- [] Support of other frameworks than PyTorch
- [] Add first comparsion based on CIFAR with a small CNN
  