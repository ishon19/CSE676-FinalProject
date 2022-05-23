# Deep Learning (CSE-676) Spring 2022 Group Project 

## Introduction
This is the repo for the Deep Learning group project. We have implemented <a href="https://arxiv.org/pdf/1611.07004.pdf" target="_blank">Pix2Pix</a> conditional adversarial network using Pytorch and have used different datasets to test and validate our implementation, details mentioned below.

## Setup 
For running the project locally, we need following 
* Python
* Installed instance of [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional, but essential for training huge datasets requiring more compute)

## Getting Started 
To run the project, you can choose a Pix2Pix specific dataset or create your own (as done in the anime_to_sketch.py script we created). The links to the datasets we have used in this project are listed down below:
- [Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz)
- [Anime Sketch](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)

We assume the Maps dataset is already downloaded for the purpose of this project, and have defaulted to Maps dataset if no specific dataset is provided in the command line argument as mentioned below

### How to run
Please follow below command line arguments for running the project. In general, following flags are available to use
- ```--flip``` to be used when we want to use the flip side of the image as input and other as target, ```false``` is the default value
- ```--mode``` specifies in which mode the script is starting the available options are either ```test``` or ```train``` with ```test``` being the default option in case not specified
- ```--epochs``` specifies how many epochs the training should run, defaults to ```50```
- ```--loadmodel``` specifies whether to load the saved model for futher training, defaults to ```false```

For instance to run the training on anime dataset while loading the saved model, in training mode with 100 epochs, we can use below command
```
python3 Pix2Pix.py --epochs=100 --mode=train --loadmodel=true --modelname=anime --flip=true
```
Please note that the order of the flags does not matter.

## Results
![image](https://user-images.githubusercontent.com/8505291/169737757-b9ed2887-64d9-405f-9b33-fe47375debac.png)
![image](https://user-images.githubusercontent.com/8505291/169737835-59077482-21d0-4a6d-b1bd-70bb1fa80c63.png)

## References
[1] Lecture Slides.

[2] Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A, "Image-to-image translation with conditional adversarial networks", pp. 1125-1134, 2017

[3] Mirza, Mehdi and Osindero, Simon, "Conditional generative adversarial nets", arXiv preprint arXiv:1411.1784, 2014

[4] Zhang, Richard and Isola, Phillip and Efros, Alexei A, "Colorful image colorization", European conference on computer vision, pp. 649-666, Springer, 2016
