from time import strftime
print(strftime('%l:%M%p'), "Begin preprocessing.", flush = True)

import os
import mne
import numpy as np
# Load data:
DATA_FOLDER = 'data/BCI_Comp_III_Wads_2004'
OUTPUT_FOLDER = 'data/preprocessed'
FEATURE_LENGTH = 700

NUM_CHANNELS = 64
SAMPLING_FREQ = 240

VAL_SPLIT = 0.1

TEST_TEXT_A = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
# TEST_TEXT_B = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'

L_FREQ = .4
H_FREQ = 10

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


# Data visualization:

def plot_epochs(data, num_epochs):
    info = mne.create_info(NUM_CHANNELS, SAMPLING_FREQ)
    temp = mne.EpochsArray(data[:num_epochs], info)
    temp.plot(picks = 'all', show_scrollbars = False, events = True)

# plot_epochs(train_data, 5)


# freq, psd = plt.psd(train_data[0][0], Fs = SAMPLING_FREQ, noverlap = 50)
# plt.show()


# Pre-processing:

def preprocessing(data):
    info = mne.create_info(NUM_CHANNELS, SAMPLING_FREQ)
    epochs = mne.EpochsArray(data, info)

    epochs = epochs.filter(l_freq = L_FREQ, h_freq = H_FREQ, picks = 'all', n_jobs = -1)
    return epochs.get_data()

train_data = preprocessing(train_data)
test_data = preprocessing(test_data)
print(strftime('%l:%M%p'), "Preprocessing completed.", flush = True)
# plot_epochs(train_data, 5)

# freq, psd = plt.psd(train_data[0][0], Fs = SAMPLING_FREQ, noverlap = 50)
# plt.show()

choose_val = np.random.choice([0, 1], size = train_data.shape[0], p = [1 - VAL_SPLIT, VAL_SPLIT]).astype('bool')
choose_train = np.logical_not(choose_val)
train_data, val_data = train_data[choose_train], train_data[choose_val]
train_labels, val_labels = train_labels[choose_train], train_labels[choose_val]
print(strftime('%l:%M%p'), "Train/validation split complete.", flush = True)

# Save data
np.save(os.path.join(DATA_FOLDER, 'train_data.npy'), train_data)
np.save(os.path.join(DATA_FOLDER, 'train_labels.npy'), train_labels)
np.save(os.path.join(DATA_FOLDER, 'val_data.npy'), val_data)
np.save(os.path.join(DATA_FOLDER, 'val_labels.npy'), val_labels)
np.save(os.path.join(DATA_FOLDER, 'test_data.npy'), test_data)
np.save(os.path.join(DATA_FOLDER, 'test_codes.npy'), test_codes)
