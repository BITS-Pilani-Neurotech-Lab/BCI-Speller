#!/usr/bin/env python
# coding: utf-8

# # BCI Competition III Dataset 2 analysis

# In[1]:


from time import strftime
print(strftime('%l:%M%p'), "Begin experiment.", flush = True)

import os
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from matplotlib import pyplot as plt


# ## Load data:

# In[2]:


DATA_FOLDER = 'data/BCI_Comp_III_Wads_2004'
FEATURE_LENGTH = 700

NUM_CHANNELS = 64
SAMPLING_FREQ = 240

VAL_SPLIT = 0.1

TEST_TEXT_A = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
# TEST_TEXT_B = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'

L_FREQ = .4
H_FREQ = 10

matrix = [
    'abcdef',
    'ghijkl',
    'mnopqr',
    'stuvwx',
    'yz1234',
    '567879_'
]

matrix_t = [''.join(i) for i in zip(*matrix)]


# In[3]:


def extract_features(signal, flashing, stim):
    signal = signal.astype('double').reshape(-1, NUM_CHANNELS, signal.shape[1])
    signal = signal.swapaxes(1, 2).reshape(-1, NUM_CHANNELS)
    flashing = flashing.T.reshape(-1)
    stim = stim.T.reshape(-1)

    data = []
    labels = []
    sample_length = (FEATURE_LENGTH * SAMPLING_FREQ) // 1000

    for i in range(len(signal)):
        if flashing[i] and (i == 0 or not flashing[i - 1]) and i + sample_length <= len(signal):
            data.append(signal[i: i + sample_length])
            labels.append(stim[i])

    data = np.asarray(data)
    labels = np.asarray(labels)

    data = data.swapaxes(1, 2)

    return data, labels

def load_data(subject):
    train_signal = np.loadtxt(os.path.join(DATA_FOLDER, f'{subject}_Train_Signal.txt'))
    train_flashing = np.loadtxt(os.path.join(DATA_FOLDER, f'{subject}_Train_Flashing.txt'))
    train_stim_type = np.loadtxt(os.path.join(DATA_FOLDER, f'{subject}_Train_StimulusType.txt'))

    test_signal = np.loadtxt(os.path.join(DATA_FOLDER, f'{subject}_Test_Signal.txt'))
    test_flashing = np.loadtxt(os.path.join(DATA_FOLDER, f'{subject}_Test_Flashing.txt'))
    test_stim_code = np.loadtxt(os.path.join(DATA_FOLDER, f'{subject}_Test_StimulusCode.txt'))

    train_data, train_labels = extract_features(train_signal, train_flashing, train_stim_type)
    test_data, test_codes = extract_features(test_signal, test_flashing, test_stim_code)

    return train_data, train_labels, test_data, test_codes

train_data, train_labels, test_data, test_codes = load_data('Subject_A')
test_text = TEST_TEXT_A
print(strftime('%l:%M%p'), "Data loaded.", flush = True)


# ## Data visualization:

# In[4]:


def plot_epochs(data, num_epochs):
    info = mne.create_info(NUM_CHANNELS, SAMPLING_FREQ)
    temp = mne.EpochsArray(data[:num_epochs], info)
    temp.plot(picks = 'all', show_scrollbars = False, events = True)

# plot_epochs(train_data, 5)


# In[5]:


# freq, psd = plt.psd(train_data[0][0], Fs = SAMPLING_FREQ, noverlap = 50)
# plt.show()


# ## Pre-processing:

# In[6]:


def preprocessing(data):
    info = mne.create_info(NUM_CHANNELS, SAMPLING_FREQ)
    epochs = mne.EpochsArray(data, info)

    epochs = epochs.filter(l_freq = L_FREQ, h_freq = H_FREQ, picks = 'all', n_jobs = -1)
    return epochs.get_data()

train_data = preprocessing(train_data)
test_data = preprocessing(test_data)
print(strftime('%l:%M%p'), "Preprocessing completed.", flush = True)
# plot_epochs(train_data, 5)


# In[7]:


# freq, psd = plt.psd(train_data[0][0], Fs = SAMPLING_FREQ, noverlap = 50)
# plt.show()


# In[8]:


choose_val = np.random.choice([0, 1], size = train_data.shape[0], p = [1 - VAL_SPLIT, VAL_SPLIT]).astype('bool')
choose_train = np.logical_not(choose_val)
train_data, val_data = train_data[choose_train], train_data[choose_val]
train_labels, val_labels = train_labels[choose_train], train_labels[choose_val]
print(strftime('%l:%M%p'), "Train/validation split complete.", flush = True)


# ## Train LDA:

# In[9]:


def train_lda(train_data, train_labels):
    train_data_copy = train_data.reshape(train_data.shape[0], -1)

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_data_copy, train_labels)

    return lda

# lda = train_lda(train_data, train_labels)


# In[10]:


def make_predictions(lda, val_data, val_labels, test_data, test_codes, text):
    val_data_copy = val_data.reshape(val_data.shape[0], -1)
    val_acc = lda.score(val_data_copy, val_labels)

    test_data_copy = test_data.reshape(test_data.shape[0], -1)
    predictions = lda.predict_proba(test_data_copy)

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

    return val_acc, test_acc

# val_acc, test_acc = make_predictions(lda, val_data, val_labels, test_data, test_codes, test_text)
# print("Complete electrode set.")
# print(f'Val Accuracy: {val_acc * 100:.2f}%')
# print(f'Test Accuracy: {test_acc * 100:.2f}%')
# print(strftime('%l:%M%p'), flush = True)


# # Compiled pipeline:

# In[11]:


def get_accuracy(channels):
    channels = sorted(channels)
    sel_train_data = train_data[:, channels, :]
    sel_val_data = val_data[:, channels, :]
    sel_test_data = test_data[:, channels, :]

    lda = train_lda(sel_train_data, train_labels)
    return make_predictions(lda, sel_val_data, val_labels, sel_test_data, test_codes, test_text)

# val_acc, test_acc = get_accuracy({0, 5, 7})
# print("{0, 5, 7} electrodes")
# print(f'Val Accuracy: {val_acc * 100:.2f}%')
# print(f'Test Accuracy: {test_acc * 100:.2f}%')
# print(strftime('%l:%M%p'), flush = True)


# In[12]:


def srs(acc_metric, max_channels):
    channels = [set() for _ in range(NUM_CHANNELS)]
    best_acc = [-1] * NUM_CHANNELS

    for count in range(max_channels):
        prev_channels = set()
        if count > 0:
            prev_channels = channels[count - 1]

        for i in range(NUM_CHANNELS):
            if i in prev_channels:
                continue

            curr_channels = prev_channels.copy()
            curr_channels.add(i)

            acc = get_accuracy(curr_channels)
            if acc[acc_metric] > best_acc[count]:
                best_acc[count] = acc[acc_metric]
                channels[count] = curr_channels

        print(strftime('%l:%M%p'), f'Electrodes: {count + 1}; Accuracy: {best_acc[count] * 100:.2f}%', flush = True)
    return channels, best_acc


# In[ ]:


speller_channels, speller_accuracies = srs(1, NUM_CHANNELS)
print(*speller_channels, sep = '\n')
print(speller_accuracies, flush = True)
# plt.plot(speller_accuracies)
# plt.show()


# In[13]:


p300_channels, p300_accuracies = srs(0, NUM_CHANNELS)
print(*p300_channels, sep = '\n')
print(p300_accuracies, flush = True)
# plt.plot(p300_accuracies)
# plt.show()

