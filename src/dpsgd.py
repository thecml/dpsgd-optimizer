import numpy as np
from numpy import random
import tensorflow as tf
import collections
import os
import time
import matplotlib.pyplot as plt
from accountant import *
from sanitizer import *

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
IMAGE_SIZE = 28
BATCH_SIZE = 64
LEARNING_RATE = 0.01
L2NORM_BOUND = 4.0
SIGMA = 4.0
N_CHANNELS = 1
DATASET = 'mnist'
MODEL_TYPE = 'dense'
USE_PRIVACY = False

def load_mnist():
    image_size = 28
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, (-1, image_size, image_size))
    X_test = np.reshape(X_test, (-1, image_size, image_size))
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, y_train, X_test, y_test
        
def load_cifar10():
    image_size = 32
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = np.reshape(X_train, (-1, image_size, image_size))
    X_test = np.reshape(X_test, (-1, image_size, image_size))
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, y_train, X_test, y_test    
 
def main():
    if DATASET == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist()
    else:
        X_train, y_train, X_test, y_test = load_cifar10()
        
    # Set aside valid set for training
    X_valid = X_train[-10000:]
    y_valid = y_train[-10000:]
    X_train = X_train[:-10000]
    y_train = y_train[:-10000]
    
    # Prepare network
    if MODEL_TYPE == 'dense':
        model = make_dense_model((IMAGE_SIZE, IMAGE_SIZE))
    else:
        model = make_cnn_model((IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(LEARNING_RATE)

    # Set constants for this loop
    eps = 1.0
    delta = 1e-7
    max_eps = 64.0 #8.0
    max_delta = 1e-3
    target_eps = [16.0] #8.0
    target_delta = [1e-5] #unused
    
    # Create accountant, sanitizer and metrics
    accountant = AmortizedAccountant(len(X_train))
    sanitizer = AmortizedGaussianSanitizer(accountant, [L2NORM_BOUND / BATCH_SIZE, True])
    mean_loss = tf.keras.metrics.Mean()
    train_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    valid_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    
    # Run training loop
    start_time = time.time()
    spent_eps_delta = EpsDelta(0, 0)
    should_terminate = False
    n_epochs = 100
    n_steps = len(X_train) // BATCH_SIZE
    for epoch in range(1, n_epochs +1):
        if should_terminate:
            spent_eps = spent_eps_delta.spent_eps
            spent_delta = spent_eps_delta.spent_delta
            print(f"Used privacy budget for {spent_eps:.4f}" +
                   f" eps, {spent_delta:.8f} delta. Stopping ...")
            break
        print(f"Epoch {epoch}/{n_epochs}")
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            if USE_PRIVACY:
                sanitized_grads = []
                eps_delta = EpsDelta(eps, delta)
                for px_grad in gradients:
                    sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, SIGMA)
                    sanitized_grads.append(sanitized_grad)
                spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]
                optimizer.apply_gradients(zip(sanitized_grads, model.trainable_variables))
                if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                    should_terminate = True
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            mean_loss(loss)
            for metric in train_metrics:
                metric(y_batch, y_pred)
            if step % 200 == 0:
                time_taken = time.time() - start_time
                for metric in valid_metrics:
                    X_batch, y_batch = random_batch(X_valid, y_valid)
                    y_pred = model(X_batch, training=False)
                    metric(y_batch, y_pred)
                if USE_PRIVACY:
                        print_status_bar(step * BATCH_SIZE, len(y_train), mean_loss, time_taken,
                                         train_metrics + valid_metrics, spent_eps_delta,) 
                else:
                    print_status_bar(step * BATCH_SIZE, len(y_train), mean_loss, time_taken,
                                     train_metrics + valid_metrics)
            if should_terminate:
                break
            
def make_cnn_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def make_dense_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def random_batch(X, y, batch_size=64):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(iteration, total, loss, time_taken, metrics=None, spent_eps_delta=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    spent_eps = spent_eps_delta.spent_eps
    spent_delta = spent_eps_delta.spent_delta
    print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
           f"{spent_eps:.4f}" + " - spent delta: " + f"{spent_delta:.8f}"
           " - time spent: " + f"{time_taken}" "\n", end=end)

if __name__ == "__main__":
    main()