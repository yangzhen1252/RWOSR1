# CIOSR
Causal Inference-inspired Open Set Recognition.

This is a description of the source code for "Open Set Recognition in Real World".

<img src='1.jpg' width=80%>


## Introduction
1. A new open-set recognition task, RWOSR, was introduced for the purpose of
addressing open-set recognition challenges in the real world.
2. A novel Causal Inference-inspired OSR (CIOSR) method has been proposed to
address the problem of OSR in real-world scenarios. With our proposed method, we
have successfully tackled the challenges of both covariate shift and semantic shift in
RWOSR, thereby enhancing the feasibility of applying OSR in real-world settings.

## Installation
### Requirements
- Linux with Python >= 3.6
- [PyTorch](https://pytorch.org/) >= 1.1.0
- torchvision >= 0.3.0
- tensorboard >= 1.14.0

## Quick Start
### Train CIOSR
```bash
python RWOSR_train --gpu 0
```
### Test CIOSR

```bash
python OSR1 --gpu 0
```
Result files will be saved in `results/`.

The training and testing settings of CIOSR are displayed in 
```bash
ops/config.py
```

The trained models can be  obtained in [CIOSR-Resnet18](https://drive.google.com/file/d/1Pg7DqU05C4QvfhobDkmyTE-tUXibcIsm/view?usp=drive_link),[CIOSR-Resnet50](https://drive.google.com/file/d/1r1oKeP9qBDYLpELFrAX5MskSe9PZncL6/view?usp=drive_link),[CIOSR-Resnet34](https://drive.google.com/file/d/1Pg7DqU05C4QvfhobDkmyTE-tUXibcIsm/view?usp=drive_link).

