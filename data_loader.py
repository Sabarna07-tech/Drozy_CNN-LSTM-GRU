import scipy.io as sio
import numpy as np

def load_data():
    filename = '/kaggle/input/eeg-driver-drowsiness-dataset/dataset (2).mat'
    data = sio.loadmat(filename)
    xdata = np.array(data['EEGsample'])
    label = np.array(data['substate'])
    subIdx = np.array(data['subindex'])
    label = label.astype(int)
    subIdx = subIdx.astype(int)
    return xdata, label, subIdx