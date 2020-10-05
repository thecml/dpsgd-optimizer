import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import collections
import os
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export
from accountant import GaussianMomentsAccountant

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
IMAGE_SIZE = 28
TARGET_EPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

def make_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
    ])
    return model

class DummySanitizer(object):
  """An sanitizer that does no sanitizing."""

  def sanitize(self, gradients, eps_delta, sigma):
      return gradients

def make_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
    ])
    return model

#@keras_export("keras.optimizers.DPSGD")
class DPSGD_Optimizer(tf.optimizers.SGD):
    """Differentially private gradient descent optimizer."""
    def __init__(self, learning_rate, accountant, sanitizer, use_locking=False, name="DPSGD_Optimizer"):
        super(DPSGD_Optimizer, self).__init__(learning_rate, use_locking, name)
        self._accountant = accountant
        self._sanitizer = sanitizer

    def minimize(self, loss, weights, batch_size, eps_delta, sigma, tape):
        priv_accum_op = self._accountant.accumulate_privacy_spending(eps_delta, sigma, batch_size)
        with tf.control_dependencies(priv_accum_op):
            gradients = tape.gradient(loss, weights)
            sanitized_grad = self._sanitizer.sanitize(gradients, eps_delta, sigma)
            return self.apply_gradients(zip(gradients, weights))

def main():
    # Make model and loss_fn
    model = make_model()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Prepare the training dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, IMAGE_SIZE*IMAGE_SIZE))
    x_train /= 255
    x_test = np.reshape(x_test, (-1, IMAGE_SIZE*IMAGE_SIZE))
    x_test /= 255
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Run training loop
    epochs = 5
    sigma = 4
    max_eps = 2.0
    max_delta = 1e-05
    eps_delta = EpsDelta(np.inf, 1.0)
    
    accountant = GaussianMomentsAccountant(len(x_train))
    sanitizer = DummySanitizer()
    dp_opt = DPSGD_Optimizer(0.01, accountant, sanitizer)
    
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                eps, delta = (0, 0)
                max_target_eps = max(target_eps)
                while eps <= max_eps and delta <= max_delta:
                    dp_opt.minimize(loss_value, model.weights, batch_size, eps_delta, sigma, tape)
                    spent_eps_deltas = accountant.get_privacy_spent(target_eps=max_target_eps)[0]
                print(f'Final epsilon: {eps}')
                print(f'Final delta: {delta}')
                
                if step % 200 == 0:
                    print(f"Training loss at step: {step}: {float(loss_value)}")
                    print(f"So far trained on {(step+1) * 64} samples")
            
            
if __name__ == "__main__":
    main()