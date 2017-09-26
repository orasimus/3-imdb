import numpy as np
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
from keras.datasets import imdb
import matplotlib.pyplot as plt
import os


def train():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # word_index is a dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    # We reverse it, mapping integer indices to words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # We decode the review; note that our indices were offset by 3
    # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

    # Our vectorized training data
    x_train = vectorize_sequences(train_data)
    # Our vectorized test data
    x_test = vectorize_sequences(test_data)

    # Our vectorized labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]


    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    return model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


def plot(history, output_dir):
    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss_values, 'bo')
    # b+ is for "blue crosses"
    plt.plot(epochs, val_loss_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(output_dir, 'loss.png'))

    plt.clf() # clear figure
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']

    plt.plot(epochs, acc_values, 'bo')
    plt.plot(epochs, val_acc_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.savefig(os.path.join(output_dir, 'accuracy.png'))


if __name__ == '__main__':
    history = train()
    output_dir = os.getenv('VH_OUTPUTS_DIR', '.')
    plot(history, output_dir)
