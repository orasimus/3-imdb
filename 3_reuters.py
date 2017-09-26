from keras import layers
from keras import models
from keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np
import os


def train():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # Note that our indices were offset by 3
    # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

    # Our vectorized training data
    x_train = vectorize_sequences(train_data)
    # Our vectorized test data
    y_train = vectorize_sequences(test_data)

    # Our vectorized training labels
    one_hot_train_labels = to_one_hot(train_labels)
    # Our vectorized test labels
    one_hot_test_labels = to_one_hot(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    return model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# from keras.utils.np_utils import to_categorical
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


def plot(history, output_dir):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo')
    plt.plot(epochs, val_loss_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(output_dir, 'accuracy.png'))


if __name__ == '__main__':
    history = train()
    output_dir = os.getenv('VH_OUTPUTS_DIR',  '.')
    plot(history, output_dir)
