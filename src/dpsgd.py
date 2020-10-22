import numpy as np
import tensorflow as tf
import collections
import time
import matplotlib.pyplot as plt
from accountant import *
from sanitizer import *
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
MNIST_SIZE = 28
CIFAR10_SIZE = 32
BATCH_SIZE = 64
LEARNING_RATE = 0.01
L2NORM_BOUND = 4.0
SIGMA = 4.0
DATASET = 'mnist'
MODEL_TYPE = 'cnn'
USE_PRIVACY = True
PLOT_RESULTS = True
N_EPOCHS = 100

def load_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], MNIST_SIZE, MNIST_SIZE, 1)
    X_test = X_test.reshape(X_test.shape[0], MNIST_SIZE, MNIST_SIZE, 1)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, y_train, X_test, y_test
        
def load_cifar10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], CIFAR10_SIZE, CIFAR10_SIZE, 3)
    X_test = X_test.reshape(X_test.shape[0], CIFAR10_SIZE, CIFAR10_SIZE, 3)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, y_train, X_test, y_test    

def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 70)
    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]
    return X_train, y_train, X_test, y_test
 
def main():
    if DATASET == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist()
        num_classes = 10
        image_size = MNIST_SIZE
        n_channels = 1
    else:
        X_train, y_train, X_test, y_test = load_cifar10()
        num_classes = 10
        image_size = CIFAR10_SIZE
        n_channels = 3
            
    # Create train/valid set
    X_train, y_train, X_valid, y_valid = shuffle_split_data(X_train, y_train)
    
    # Prepare network
    if MODEL_TYPE == 'dense':
        model = make_dense_model((image_size, image_size, n_channels),
                                  image_size*image_size, num_classes)
    else:
        model = make_cnn_model((image_size, image_size, n_channels), num_classes)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(LEARNING_RATE) 

    # Set constants for this loop
    eps = 1.0
    delta = 1e-7
    max_eps = 16.0
    max_delta = 1e-3
    target_eps = [16.0]
    target_delta = [1e-5] #unused
    
    # Create accountant, sanitizer and metrics
    accountant = AmortizedAccountant(len(X_train))
    sanitizer = AmortizedGaussianSanitizer(accountant, [L2NORM_BOUND / BATCH_SIZE, True])
    
    # Setup metrics
    train_mean_loss = tf.keras.metrics.Mean()
    valid_mean_loss = tf.keras.metrics.Mean()
    train_acc_scores, valid_acc_scores = list(), list()
    train_loss_scores, valid_loss_scores = list(), list()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    valid_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    test_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    
    # Run training loop
    start_time = time.time()
    spent_eps_delta = EpsDelta(0, 0)
    should_terminate = False
    n_steps = len(X_train) // BATCH_SIZE
    for epoch in range(1, N_EPOCHS + 1):
        if should_terminate:
            spent_eps = spent_eps_delta.spent_eps
            spent_delta = spent_eps_delta.spent_delta
            print(f"Used privacy budget for {spent_eps:.4f}" +
                   f" eps, {spent_delta:.8f} delta. Stopping ...")
            break
        print(f"Epoch {epoch}/{N_EPOCHS}")
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
                train_acc_metric.update_state(y_batch, y_pred)
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
            train_mean_loss(loss)
            for metric in train_metrics:
                metric(y_batch, y_pred)
            if step % 200 == 0:
                time_taken = time.time() - start_time
                for metric in valid_metrics:
                    X_batch, y_batch = random_batch(X_valid, y_valid)
                    y_pred = model(X_batch, training=False)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                    valid_mean_loss(loss)
                    valid_acc_metric.update_state(y_batch, y_pred)
                    metric(y_batch, y_pred)
                if USE_PRIVACY:
                    print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics, spent_eps_delta,) 
                else:
                    print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics)
            if should_terminate:
                break
            
        # Update training scores
        train_acc = train_acc_metric.result()
        train_loss = train_mean_loss.result()
        train_acc_scores.append(train_acc)
        train_loss_scores.append(train_loss)
        train_acc_metric.reset_states()
        train_mean_loss.reset_states()
        
        # Update validation scores
        valid_acc = valid_acc_metric.result()
        valid_loss = valid_mean_loss.result()
        valid_acc_scores.append(valid_acc)
        valid_loss_scores.append(valid_loss)
        valid_acc_metric.reset_states()
        valid_mean_loss.reset_states()
    
    # Evaluate model
    for metric in test_metrics:
        y_pred = model(X_test, training=False)
        metric(y_test, y_pred)
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in test_metrics or []])
    print(f"Training completed, test metrics: {metrics}")
    
    # Save model
    version = "DPSGD" if USE_PRIVACY else "SGD"
    model.save(MODELS_DIR/f"{version}-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.h5")
    
    # Make plots
    if PLOT_RESULTS:
        epochs_range = range(1, N_EPOCHS+1)
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_loss_scores, color='blue', label='Training loss')
        plt.plot(epochs_range, valid_loss_scores, color='red', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Loss-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()
         
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_acc_scores, color='blue', label='Training accuracy')
        plt.plot(epochs_range, valid_acc_scores, color='red', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Accuracy-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()

def make_dense_model(input_shape, units, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def make_cnn_model(input_shape, num_classes):
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
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def random_batch(X, y, batch_size=64):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(iteration, total, loss, time_taken, metrics=None, spent_eps_delta=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    if spent_eps_delta:
        spent_eps = spent_eps_delta.spent_eps
        spent_delta = spent_eps_delta.spent_delta
        print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
               f"{spent_eps:.4f}" + " - spent delta: " + f"{spent_delta:.8f}"
               " - time spent: " + f"{time_taken}" "\n", end=end)
    else:
        print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
              " - time spent: " + f"{time_taken}" "\n", end=end)

if __name__ == "__main__":
    main()