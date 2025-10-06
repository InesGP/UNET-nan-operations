# NaN-Aware Operations

This repository contains code and experiments for **NaN Unpooling & Pooling** a.k.a Conservative & Aggressive NaNs, novel PyTorch-based operations designed to accelerate convolutional neural network (CNN) inference by skipping computations on numerically irrelevant data. These techniques were developed as part of research on numerical uncertainty in CNNs for neuroimaging, where we observed that a substantial fraction of operations in standard CNNs are applied to numerical noise, contributing little to no effect on outputs.


## Features

- **NaN Unpooling & Pooling operations** implemented in [`nan_ops.py`](https://github.com/InesGP/UNET-nan-operations/blob/main/nan_ops.py).
- Scripts and notebooks for:
  - Exploring original max pooling bug that causes large uncertainty in FastSurfer + initial NaN operations ([`NaN_Operations_Exploration.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/NaN_Operations_Exploration.ipynb))
  - Aggressive NaN analysis for FastSurfer and FONDUE use cases ([`Pooling_Analysis.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/Pooling_Analysis.ipynb))
  - Conservative NaN analysis for FastSurfer and FONDUE use cases ([`Unpool_Analysis.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/Unpool_Analysis.ipynb))
  - Aggressive NaN analysis for MNIST use case ([`mnist_usecase/NaN_MNIST.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/mnist_usecase/NaN_MNIST.ipynb))
  - Aggressive NaN analysis for Xception use case ([`depth_sep_conv/Xception.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/depth_sep_conv/Xception.ipynb))
  - Time trial experiments comparing NaN convolutions with standard convolutions ([`time_trials/Time_Trials.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/time_trials/Time_Trials.ipynb)) 
- Listed files contain the primary analyses, while other files include secondary analyses and instructions for running experiments and obtaining derived results.
- Unit tests ([`nan_unit_test.py`](https://github.com/InesGP/UNET-nan-operations/blob/main/nan_unit_test.py))
