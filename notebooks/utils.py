"""Training helpper functions
"""

"""Plot accuracy for train and val datasets and also loss function vs number of epoch
"""

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [100, 30]


def plot_report(histories, metric='accuracy'):
    legend_pairs = [['Train_{}'.format(key), 'Val_{}'.format(key)] for key in histories]
    legend = [item for sublist in legend_pairs for item in sublist]
    
    for key in histories:
        history=histories[key]
        # Plot training & validation loss values
        plt.plot(history.history[metric])
        plt.plot(history.history['val_{}'.format(metric)][1:], '--')

    plt.title('Model {}'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left', bbox_to_anchor=(1.04, 1))

    
def plot_report_t(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()