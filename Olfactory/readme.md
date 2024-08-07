# Spiking olfactory



## Introduction

Real-time olfactory detection using ANNs and SNNs, combining the respective advantages of ANNs and SNNs to achieve high-precision performance.

## Dataset

The dataset needs to be constructed by you.

## How to construct a dataset?

First of all, you need to use ardiuno to run the "Olfactory.ino" file in the folder, and use "readtxt.py" to accept the data from ardiuno, and then run "toexcel.py" to convert the txt file into an excel file, and you have a data set of your own!
## Get Started


```
cd Olfactory
start Olfactory.ino
python readtxt.py
python toexcel.py
python main.py 
```

The default time step T=4, if you want to change the time step and other parameters, please go to the corresponding place in the model.py or main.py file to change it

## Real-time detection
```
cd Haptic
python dynamic_classes.py
```

## Pre-trained models

link: https://pan.baidu.com/s/16FmN_HOKd3qUsa1sMVKHqw 

extraction code: g11h



