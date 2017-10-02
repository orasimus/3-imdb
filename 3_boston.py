from keras.datasets import boston_housing
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import numpy as np


def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    return model


def train():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
        # Prepare the validation data: data from partition # k
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # Prepare the training data: data from all other partitions
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        # Build the Keras model (already compiled)
        model = build_model()
        # Train the model (in silent mode, verbose=0)
        history = model.fit(partial_train_data, partial_train_targets,
                  validation_data=(val_data, val_targets),
                  epochs=num_epochs, batch_size=1, verbose=0)
        # Evaluate the model on the validation data
        # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(val_mae)

    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    return average_mae_history


def plot(average_mae_history, output_dir):
    plt.plot(range(len(average_mae_history) - 10), average_mae_history[10:])
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.savefig(os.path.join(output_dir, 'mae.png'))


if __name__ == '__main__':
    average_mae_history = train()
    output_dir = os.getenv('VH_OUTPUTS_DIR',  '.')
    plot(average_mae_history, output_dir)
