# Differentially Private Stochastic Gradient Descent

This is a partly implementation of the differentially private SGD optimizer described in the [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) paper using a Gaussian sanitizer to sanitize gradients and a amortized accountant to keep track of used privacy. AmortizedGaussianSanitizer sanitizes gradients with Gaussian noise in an amoritzed way. AmortizedAccountant accumulates the privacy spending by assuming all the examples are processed uniformly at random, so the spending is amortized among all the examples. Implementation is done in Tensorflow 2.3.

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
| DPSGD-DENSE|  47.14%     | 47.37%      | 48.89%    | 13.99    | 0.00036839 | 14M 52S
| DPSGD-CNN  |  67.35%     | 67.68%      | 71.06%    | 2.29     | 0.00012746 | 52M 15S

Table 2. results of 200 epochs training with the learning rate of 0.01

| Model      | Train acc.  | Valid acc.  | Test acc. | Eps used | Delta used | Training time |
| -----      | -----       | ----        | ----      | ----     | ----       | ----
| DPSGD-DENSE|  47.93%     | 48.11%      |  49.14%   | 19.79    | 0.00073558 | 28M 48S
| DPSGD-CNN  |  73.97%     | 74.22%      |  76.83%   | 3.23     | 0.00024880 | 1H 46M

The accuracy for DPSGD-CNN on MNIST for 200 epochs
<img src="https://raw.githubusercontent.com/thecml/dpsgd-optimizer/master/results/DPSGD-Accuracy-200-cnn-mnist.png" width="640" height="480">

The loss for DPSGD-CNN on MNIST for 200 epochs
<img src="https://raw.githubusercontent.com/thecml/dpsgd-optimizer/master/results/DPSGD-Loss-200-cnn-mnist.png" width="640" height="480">

## Acknowledgements
Acknowledgements given to [marcotcr](https://github.com/marcotcr/tf-models).

## References
Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (ACM CCS), pp. 308-318, 2016.

