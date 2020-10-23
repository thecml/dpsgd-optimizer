# Differentially Private Stochastic Gradient Descent

This is a implementation of the differentially private SGD optimizer described in the [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) paper using a Gaussian sanitizer to sanitize gradients and a amortized accountant to keep track of used privacy. AmortizedGaussianSanitizer sanitizes gradients with Gaussian noise in an amoritzed way. AmortizedAccountant accumulates the privacy spending by assuming all the examples are processed uniformly at random, so the spending is amortized among all the examples. Implementation is done in Tensorflow 2.3.

Note: The scripts will be slow without CUDA enabled.

## Requirements
python>=3.6
tensorflow>=2.0

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters eps=1.0, delta=1e-7, target_eps=16. For DENSE network, we used a max_eps=16 and max_delta=1e-3. For CNN network, we used max_eps=64, max_delta=1e-3.

Table 1. results of 100 epochs training with the learning rate of 0.01

| Model      | Train acc.  | Valid acc.  | Test acc. | Eps used | Delta used | Training time |
| -----      | -----       | ----        | ----      | ----     | ----       | ----
| DPSGD-DENSE|  xx.xx%     | xx.xx%      |           | 8.29     | 0.00016385 | 20.1 minutes
| DPSGD-CNN  |  xx.xx%     | xx.xx%      |  xx.xx%   | 3.23     | 0.00024880 | 94.95 minutes

Table 2. results of 200 epochs training with the learning rate of 0.01

| Model      | Train acc.  | Valid acc.  | Test acc. | Eps used | Delta used | Training time |
| -----      | -----       | ----        | ----      | ----     | ----       | ----
| DPSGD-DENSE|  xx.xx%     | xx.xx%      |           | x.xx     | 0.00016385 | 20.1 minutes
| DPSGD-CNN  |  74.19%     | 74.62%      |  81.86%   | 3.23     | 0.00024880 | 94.95 minutes


## Acknowledgements
Acknowledgements given to [marcotcr](https://github.com/marcotcr/tf-models).

## References
Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (ACM CCS), pp. 308-318, 2016.

