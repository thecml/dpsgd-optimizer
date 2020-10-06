import numpy as np
import tensorflow as tf
import collections
import os
import time
from accountant import GaussianMomentsAccountant
import matplotlib.pyplot as plt

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
IMAGE_SIZE = 28
N_CHANNELS = 1
BATCH_SIZE = 64
C = 4.0

class GaussianSanitizer(object):
  def sanitize(self, gradients, sigma):
      gradients = [tf.clip_by_norm(g, clip_norm=C) for g in gradients]
      gradients += np.random.normal(0, (sigma ** 2)*(C ** 2), len(gradients))
      return gradients
    
class DPSGD_Optimizer(tf.optimizers.SGD):
    def __init__(self, learning_rate, accountant, sanitizer, use_locking=False, name="DPSGD_Optimizer"):
        super(DPSGD_Optimizer, self).__init__(learning_rate, use_locking, name)
        self._accountant = accountant
        self._sanitizer = sanitizer

    def minimize(self, gradients, weights, eps_delta, sigma):
        self._accountant.accumulate_privacy_spending(eps_delta, sigma, BATCH_SIZE)
        gradients[1] = self._sanitizer.sanitize(gradients[1], sigma)
        return self.apply_gradients(zip(gradients, weights))

def make_model(input_shape):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                  kernel_initializer='he_uniform', input_shape=input_shape))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	return model

def main():
    # Prepare the training and test dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))
    x_train = x_train.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))
    x_test = x_test.astype("float32") / 255.0

    # Prepare valid dataset.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
    
    # Prepare network
    model = make_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Set constants for this loop
    epochs = 10
    sigma = 4.0
    max_eps = 1
    max_delta = 1e-03
    eps_delta = EpsDelta(np.inf, 1.0)
    target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    target_delta = [1e-1]
    total_examples = len(x_train)
    use_privacy = True
    
    # Create objects
    accountant = GaussianMomentsAccountant(total_examples)
    sanitizer = GaussianSanitizer()
    dp_opt = DPSGD_Optimizer(0.01, accountant, sanitizer)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_scores, valid_scores = list(), list()
    train_loss, valid_loss = list(), list()
    
    # Run training loop
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()
        number_steps = int(total_examples / BATCH_SIZE)
        step_max_eps = max_eps / number_steps
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            if int(step) == 0:
                train_loss.append(loss_value.numpy()) # Save loss for this epoch
            gradients = tape.gradient(loss_value, model.trainable_weights)
            spent_eps_deltas = EpsDelta(0, 0)
            if use_privacy:
                while spent_eps_deltas.spent_eps <= step_max_eps and spent_eps_deltas.spent_delta <= max_delta:
                    dp_opt.minimize(gradients, model.trainable_weights, eps_delta, sigma)
                    spent_eps_deltas = accountant.get_privacy_spent(target_deltas=target_delta)[0]
            else:
                tf.optimizers.SGD().apply_gradients(zip(gradients, model.trainable_weights))
            train_acc_metric.update_state(y_batch_train, logits)    
            if step % 200 == 0:
                print(f"Training loss at step: {step}: {float(loss_value)}")
                print(f"So far trained on {(step+1) * 64} samples")
                print(f"Privacy spent: eps {spent_eps_deltas.spent_eps}, delta {spent_eps_deltas.spent_delta}")    
        
        print(f"Privacy spent: eps {spent_eps_deltas.spent_eps}, delta {spent_eps_deltas.spent_delta}")    
        train_acc = train_acc_metric.result()
        train_scores.append(train_acc)
        print(f"Training acc over epoch: {float(train_acc)}")
        train_acc_metric.reset_states()
        
        batch_valid_loss = list()
        for x_batch_valid, y_batch_valid in valid_dataset:
            valid_logits = model(x_batch_valid, training=False)
            valid_acc_metric.update_state(y_batch_valid, valid_logits)
            batch_valid_loss.append(loss_fn(y_batch_valid, valid_logits))
        valid_loss.append(np.mean(batch_valid_loss))
        
        valid_acc = valid_acc_metric.result()
        valid_acc_metric.reset_states()
        valid_scores.append(valid_acc)
        print("Validation acc: %.4f" % (float(valid_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
    
    # Show training loss
    epochs_range = range(1, len(train_loss)+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_loss, color='blue', label='Training loss')
    plt.plot(epochs_range, valid_loss, color='red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Show validation loss
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_scores, color='blue', label='Training accuracy')
    plt.plot(epochs_range, valid_scores, color='red', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()