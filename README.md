# BFU

# Bayesain Federated Unlearning and BFU-SS

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