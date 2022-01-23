# BSS
Bike Sharing System Network prediction

By Siyu Chen.

This repository is an official implementation of the report Bike-sharing Systems Network Analysis.


## Introduction

**TL; DR.** Over the past decade, bike-sharing systems (BSS) have been growing in number and popularity in cities across the world. However, it is impossible to analyze too dense graphs based on network science. Furthermore, we need to infer the new interactions among its members in the near future. 

**Abstract.** In bike-sharing systems (BSS) tasks, how to determine the weight threshold matrix and develop a predictive model for link prediction are two main challenges. In task 1, the minimum absolute spectral similarity (MASS) technique for establishing the weight threshold matrix is used to limit the number of connections and generate the resulting network while keeping the network features. In task 2, we use Neural Network (NN) based classifier to develop a model to predict the directed binary BSS network. The weighted sampler is applied to improve pre-processed dataset to fix the class imbalance between edge and non-edge classes. Finally, the optimal weighted edge to non-edge ratio is determined by analyzing the models' performance

## License

This project is released under the [Apache 2.0 license](./LICENSE).

```

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Training

#### Training of NN-network classifier

For example, the command for training CNN classifer:

```bash
python BSS_network_analysis_NN.py
```

#### Training of SVM classifier
For example, the command for training SVM classifier:

```bash
python BSS_network_analysis_SVM.py
```

### Evaluation

You can get the visualize the results by running following command :

```bash
python plot.py
```

You can also run distributed evaluation by using ```./tools/run_dist_launch.sh``` or ```./tools/run_dist_slurm.sh```.
