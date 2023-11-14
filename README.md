# BFU

# BFU: Bayesian Federated Unlearning with Parameter Self-Sharing

### Overview
This repository is the official implementation of BFU and BFU-SS, and the corresponding paper is accepted by ASIACCS23.
We only use one middle layer of the model to implement the variational unlearning.
We also upload the compared method HFU implemented by ourself.


### Prerequisites

```
python = 3.8
pytorch = 1.4.0
matplotlib
numpy
```

### Running the experiments

1. To run the BFU and BFU-SS on MNIST
```
python /BFU/BackdoorFedAvg.py
```

2. To run the BFU and BFU-SS on CIFAR10
```
python /BFU/CIFAR_5u_10e.py
```

3. To run the HFU on MNIST
```
python /HFU/backdoor_FedHessian.py
```

4. To run the HFU on CIFAR
```
python /HFU/backdoor_FedHessian_er01_cifar.py
```

### Citation

```
@inproceedings{wang2023bfu,
  title={BFU: Bayesian Federated Unlearning with Parameter Self-Sharing},
  author={Wang, Weiqi and Tian, Zhiyi and Zhang, Chenhan and Liu, An and Yu, Shui},
  booktitle={Proceedings of the 2023 ACM Asia Conference on Computer and Communications Security},
  pages={567--578},
  year={2023}
}
```