import numpy as np
import tensorflow as tf
import collections
import os
import time
import matplotlib.pyplot as plt
from accountant import GaussianMomentsAccountant

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
IMAGE_SIZE = 28
N_CHANNELS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.05
C = 4.0

class GaussianSanitizer(object):
  def sanitize(gradients, sigma):
      gradients = [tf.clip_by_norm(g, clip_norm=C) for g in gradients]
      gradients += np.random.normal(0, (sigma ** 2)*(C ** 2), len(gradients))
      return gradients
        
def make_model_cnn(input_shape):
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

def make_model_dense():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(IMAGE_SIZE*IMAGE_SIZE,)))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def main():
    # Prepare the training and test dataset.
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, IMAGE_SIZE*IMAGE_SIZE))
    x_train = x_train.astype("float32") / 255.0

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
    model = make_model_dense()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(LEARNING_RATE)
    
    # Set constants for this loop
    epochs = 10
    sigma = 5.0
    max_eps = 4.0
    max_delta = 1e-05
    eps_delta = EpsDelta(np.inf, 1.0)
    target_eps = [4.0]
    target_delta = [1e-05]
    total_examples = len(x_train)
    use_privacy = False
    
    # Create objects
    accountant = GaussianMomentsAccountant(total_examples)
    sanitizer = GaussianSanitizer()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_scores, valid_scores = list(), list()
    train_loss, valid_loss = list(), list()

    # Run training loop
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            total_loss = 0
            train_vars = model.trainable_variables
            spent_eps_deltas = EpsDelta(0, 0)
            accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
            num_samples = len(x_batch_train)
            for j in range(num_samples):
                sample = x_batch_train[j]
                sample = np.reshape(sample, (-1, IMAGE_SIZE*IMAGE_SIZE))
                with tf.GradientTape() as tape:
                    logits = model(sample, training=True)
                    loss_value = loss_fn(y_batch_train[j], logits)
                    train_acc_metric.update_state(y_batch_train, logits)
                total_loss += loss_value
                gradients = tape.gradient(loss_value, train_vars)
                if use_privacy:
                    while spent_eps_deltas.spent_eps <= max_eps and spent_eps_deltas.spent_delta <= max_delta:
                        accountant.accumulate_privacy_spending(eps_delta, sigma, BATCH_SIZE)
                        gradients = sanitizer.sanitize(gradients, sigma)
                        spent_eps_deltas = accountant.get_privacy_spent(target_eps=target_eps)[0]
                accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, gradients)]
            accum_gradient = [this_grad/num_samples for this_grad in accum_gradient]
            optimizer.apply_gradients(zip(accum_gradient, train_vars))
            if step % 200 == 0:
                num_samples = (step+1) * BATCH_SIZE
                epoch_loss = total_loss / num_samples
                print(f"So far trained on {num_samples} samples, epoch loss: {epoch_loss}")
                if use_privacy:
                    print(f"Privacy spent: eps {spent_eps_deltas.spent_eps}, delta {spent_eps_deltas.spent_delta}")    

        train_acc = train_acc_metric.result()
        train_scores.append(train_acc)
        print(f"Training acc over epoch: {float(train_acc)}")
        train_acc_metric.reset_states()
        
        batch_valid_loss = list()
        for x_batch_valid, y_batch_valid in valid_dataset:
            break
            valid_logits = model(x_batch_valid, training=False)
            valid_acc_metric.update_state(y_batch_valid, valid_logits[2])
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