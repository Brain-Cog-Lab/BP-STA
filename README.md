# Backpropagation with Biologically Plausible Spatio-Temporal Adjustment For Training Deep Spiking Neural Networks
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6489856.svg)](https://doi.org/10.5281/zenodo.6489856)

This repository contains code from our paper [**Backpropagation with Biologically Plausible Spatio-Temporal Adjustment For Training Deep Spiking Neural Networks**]. If you use our code or refer to this project, please cite it using

```
@article{shenBackpropagationBiologicallyPlausible2021,
  title = {Backpropagation with {{Biologically Plausible Spatio-Temporal Adjustment For Training Deep Spiking Neural Networks}}},
  author = {Shen, Guobin and Zhao, Dongcheng and Zeng, Yi},
  year = {2021},
  month = oct,
  journal = {arXiv:2110.08858 [cs]},
  eprint = {2110.08858},
  eprinttype = {arxiv},
  primaryclass = {cs},
  archiveprefix = {arXiv} 
}
```

## Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* torchvision


## Data preparation

First modify the ```DATA_DIR='path/to/datasets``` in ```code/datasets/__init__.py``` to the root directory of your datasets.
Neuromorphic datasets **NMNIST**, **DVS-Gesture** and **DVS-CIFAR10** need to be manually downloaded and placed under the ```/path/to/datasets/DVS/*``` 


```
/path/to/datasets/
  DVS/
    DVS_Cifar10/
    DVS_Gesture/
    NMNIST/
```

## Train

Run training scripts corresponding to different datasets.

For example, training and validating the proposed method on the MNIST dataset:

```bash 
bash ./train_dvsg.sh
```
