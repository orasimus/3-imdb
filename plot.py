import matplotlib.pyplot as plt
import os


def plot(history, output_path=''):
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, 'loss.png'))

    plt.clf()   # clear figure
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']

    plt.plot(epochs, acc_values, 'bo')
    plt.plot(epochs, val_acc_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.savefig(os.path.join(output_path, 'accuracy.png'))
