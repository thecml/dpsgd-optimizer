import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import collections
import os
import time
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
    
    # Prepare the metrics
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
    # Prepare the training dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, IMAGE_SIZE*IMAGE_SIZE))
    x_train = x_train.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, IMAGE_SIZE*IMAGE_SIZE))
    x_test = x_test.astype("float32") / 255.0

    # Prepare valid dataset.
    x_val = x_train[10000:]
    y_val = y_train[10000:]
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    valid_dataset = valid_dataset.batch(batch_size)
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
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                spent_eps_deltas = EpsDelta(0, 0)
                max_target_eps = max(TARGET_EPS)
                while spent_eps_deltas.spent_eps <= max_eps and spent_eps_deltas.spent_delta <= max_delta:
                    dp_opt.minimize(loss_value, model.weights, batch_size, eps_delta, sigma, tape)
                    spent_eps_deltas = accountant.get_privacy_spent(target_eps=TARGET_EPS)[0]
                print(f"Spent privacy: eps {spent_eps_deltas.spent_eps} delta {spent_eps_deltas.spent_delta}")
                train_acc_metric.update_state(y_batch_train, logits)
                if step % 200 == 0:
                    print(f"Training loss at step: {step}: {float(loss_value)}")
                    print(f"So far trained on {(step+1) * 64} samples")
                    
        train_acc = train_acc_metric.result()
        print(f"Training acc over epoch: {float(train_acc)}")
        train_acc_metric.reset_states()
        
        for x_batch_val, y_batch_val in valid_dataset:
            val_logits = model(x_batch_val, training=False)
            valid_acc_metric.update_state(y_batch_val, val_logits)
        valid_acc = valid_acc_metric.result()
        valid_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(valid_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        
if __name__ == "__main__":
    main()