import matplotlib.pyplot as plt
import numpy as np

def visualize_eeg_data(xdata, label, subIdx, subject=1, channel=28, sample_freq=128):
    plt.figure(figsize=(12, 6))
    subject_idx = np.where(subIdx == subject)[0]
    subject_data = xdata[subject_idx][:, channel, :]
    subject_labels = label[subject_idx]
    for i in range(min(5, subject_data.shape[0])):
        plt.plot(subject_data[i], label=f'Sample {i+1} - State: {subject_labels[i]}')
    plt.title(f'Subject {subject} - Channel {channel} EEG Signals')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True)
    plt.show()