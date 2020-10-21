# Differentially Private Stochastic Gradient Descent

This is a implementation of the differentially private SGD optimizer described in the [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) paper using an amortized accountant to keep track of used privacy. AmortizedAccountant accumulates the privacy spending by assuming all the examples are processed uniformly at random, so the spending is amortized among all the examples. Implementation is done in Tensorflow 2.3.

Note: The scripts will be slow without CUDA enabled.

## Requirements
python>=3.6
tensorflow>=2.0

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters eps=1.0, delta=1e-7, max_eps=16, max_delta=1e-3, target_eps=16

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model      | Train acc.  | Valid acc.  | Eps used | Delta used | Training time |
| -----      | -----       | ----        | ----     | ----       | ----
| DPSGD-DENSE|  94.57%     | 70.44%      |          |            |
| DPSGD-CNN  |  96.59%     | 77.72%      |          |            |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model      | Acc. of IID | Acc. of Non-DP | Eps used | Delta used | Training time |
| -----      | -----       | ----           | ----     | ----       | ----
| DPSGD-DENSE|  94.57%     | 70.44%         |          |            |
| DPSGD-CNN  |  96.59%     | 77.72%         |          |            |


## Ackonwledgements
Acknowledgements given to [marcotcr](https://github.com/marcotcr/tf-models).

## References
Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (ACM CCS), pp. 308-318, 2016.

