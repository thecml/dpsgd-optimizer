import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def make_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
    ])
    return model

def main():
    # Make model, optimizer and loss_fn
    model = make_model()
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Prepare the training dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 28*28))
    x_test = np.reshape(x_test, (-1, 28*28))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Run training loop
    epochs = 5
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            if step % 200 == 0:
                print(f"Training loss at step: {step}: {float(loss_value)}")
                print(f"So far trained on {(step+1) * 64} samples")
            
            
if __name__ == "__main__":
    main()