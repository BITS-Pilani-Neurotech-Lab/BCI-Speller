from time import strftime
print(strftime('%l:%M%p'), "Begin experiment.", flush = True)

import os
import numpy as np
# from matplotlib import pyplot as plt

from classifiers.LDA import LDA
from selection_strategies.SRS import SRS

DATA_FOLDER = 'data/preprocessed'
NUM_CHANNELS = 64
matrix = [
    'abcdef',
    'ghijkl',
    'mnopqr',
    'stuvwx',
    'yz1234',
    '567879_'
]

TEST_TEXT_A = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'

# Load data
train_data = np.load(os.path.join(DATA_FOLDER, 'train_data.npy'))
train_labels = np.load(os.path.join(DATA_FOLDER, 'train_labels.npy'))
val_data = np.load(os.path.join(DATA_FOLDER, 'val_data.npy'))
val_labels = np.load(os.path.join(DATA_FOLDER, 'val_labels.npy'))
test_data = np.load(os.path.join(DATA_FOLDER, 'test_data.npy'))
test_codes = np.load(os.path.join(DATA_FOLDER, 'test_codes.npy'))
test_text = TEST_TEXT_A

# Train classifier:
classifier = LDA()
classifier.train(train_data, train_labels)

# lda = train_lda(train_data, train_labels)

def make_predictions(classifier, val_data, val_labels, test_data, test_codes, text):
    val_predictions = classifier.predict(val_data)
    val_acc = sum(np.argmax(val_predictions, axis = 1) == val_labels) / val_labels.shape[0]

    predictions = classifier.predict(test_data)

    # Unscramble order of stimuli
    test_codes = test_codes.reshape(-1, 15, 12)
    idx = test_codes.argsort()
    static_idx = np.indices(idx.shape)

    predictions = predictions[:, 1]
    predictions = predictions.reshape(-1, 15, 12)
    predictions = predictions[static_idx[0], static_idx[1], idx]
    predictions = predictions.sum(axis = 1).argsort()

    cols = predictions[predictions <= 5].reshape(-1, 6)[:, -1]
    rows = predictions[predictions > 5].reshape(-1, 6)[:, -1]
    predicted_text = ''.join([matrix[i - 6][j] for i, j in zip(rows, cols)])
    predicted_text = predicted_text.upper()

    test_acc = sum([predicted_text[i] == text[i] for i in range(len(text))]) / len(text)

    return {'val': val_acc, 'test': test_acc}

# val_acc, test_acc = make_predictions(lda, val_data, val_labels, test_data, test_codes, test_text)
# print("Complete electrode set.")
# print(f'Val Accuracy: {val_acc * 100:.2f}%')
# print(f'Test Accuracy: {test_acc * 100:.2f}%')
# print(strftime('%l:%M%p'), flush = True)


# Compiled pipeline:
def get_accuracy(channels):
    channels = sorted(channels)
    sel_train_data = train_data[:, channels, :]
    sel_val_data = val_data[:, channels, :]
    sel_test_data = test_data[:, channels, :]

    classifier.train(sel_train_data, train_labels)
    return make_predictions(classifier, sel_val_data, val_labels, sel_test_data, test_codes, test_text)

# val_acc, test_acc = get_accuracy({0, 5, 7})
# print("{0, 5, 7} electrodes")
# print(f'Val Accuracy: {val_acc * 100:.2f}%')
# print(f'Test Accuracy: {test_acc * 100:.2f}%')
# print(strftime('%l:%M%p'), flush = True)

selection_strategy = SRS(NUM_CHANNELS, get_accuracy)
speller_channels, speller_accuracies = selection_strategy.run('val', NUM_CHANNELS)
print(*speller_channels, sep = '\n')
print(speller_accuracies, flush = True)
# plt.plot(speller_accuracies)
# plt.show()

p300_channels, p300_accuracies = selection_strategy.run('test', NUM_CHANNELS)
print(*p300_channels, sep = '\n')
print(p300_accuracies, flush = True)
# plt.plot(p300_accuracies)
# plt.show()

