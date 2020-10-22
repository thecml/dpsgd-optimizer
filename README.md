# Differentially Private Stochastic Gradient Descent

This is a implementation of the differentially private SGD optimizer described in the [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) paper using an Gaussian sanitizer to sanitize gradients and a amortized accountant to keep track of used privacy. AmortizedGaussianSanitizer sanitizes gradients with Gaussian noise in an amoritzed way. AmortizedAccountant accumulates the privacy spending by assuming all the examples are processed uniformly at random, so the spending is amortized among all the examples. Implementation is done in Tensorflow 2.3.

Note: The scripts will be slow without CUDA enabled.

## Requirements
python>=3.6
tensorflow>=2.0

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters eps=1.0, delta=1e-7, target_eps=16. For DENSE network, we used a max_eps=16 and max_delta=1e-3. For CNN network, we used max_eps=64, max_delta=1e-3.

Table 1. results of 50 epochs training with the learning rate of 0.01

| Model      | Train acc.  | Valid acc.  | Eps used | Delta used | Training time |
| -----      | -----       | ----        | ----     | ----       | ----
| DPSGD-DENSE|  50.42%     | 49.80%      | 5.86     | 0.00008159 | 9.3 minutes
| DPSGD-CNN  |  58.68%     | 58.01%      | 1.62     | 0.00006402 | 24.85 minutes

Table 2. results of 100 epochs training with the learning rate of 0.01

| Model      | Train acc.  | Valid acc.  | Eps used | Delta used | Training time |
| -----      | -----       | ----        | ----     | ----       | ----
| DPSGD-DENSE|  51.45%     | 51.45%      | 8.29     | 0.00016385 | 20.1 minutes
| DPSGD-CNN  |  67.11%     | 66.89%      | 2.29     | 0.00012746 | 50.8 minutes


## Ackonwledgements
Acknowledgements given to [marcotcr](https://github.com/marcotcr/tf-models).

## References
Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (ACM CCS), pp. 308-318, 2016.

