""" Implementation of paper H. Gamper and I. J. Tashev, "Blind Reverberation Time Estimation Using a Convolutional Neural Network," 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan, 2018, pp. 136-140, doi: 10.1109/IWAENC.2018.8521241. keywords: {Training;Estimation;Neural networks;Noise measurement;Training data;Reverberation;T60;energy decay rate;deep neural networks},

"""
import torch
import torchaudio
import numpy as np
from scipy.signal import lfilter, butter
import scipy.signal as signal
import librosa
import matplotlib.pylab as  plt
from scipy.signal import bilinear
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from log_spectral_feature_extractor import Log_spectral_feature_extraction

""" Implementation of model architecture inspired from paper H. Gamper and I. J. Tashev, "Blind Reverberation Time Estimation Using a Convolutional Neural Network," 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan, 2018, pp. 136-140, doi: 10.1109/IWAENC.2018.8521241. keywords: {Training;Estimation;Neural networks;Noise measurement;Training data;Reverberation;T60;energy decay rate;deep neural networks},
"""

class R2Dnet(nn.Module):
    def __init__(self):
        super(R2Dnet, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(1, 10), stride=(1, 2), padding=(0, 1))  # 1x10 kernel, stride (1, 2), padding (0, 1)
        # self.conv2 = nn.Conv2d(5, 5, kernel_size=(1, 10), stride=(1, 3), padding=(0, 1))  # 1x10 kernel, stride (1, 3), padding (0, 1)
        # self.conv3 = nn.Conv2d(5, 5, kernel_size=(1, 11), stride=(1, 3), padding=(0, 1))  # 1x11 kernel, stride (1, 3), padding (0, 1)
        self.conv4 = nn.Conv2d(5, 5, kernel_size=(1, 11), stride=(1, 2), padding=(0, 1))  # 1x11 kernel, stride (1, 2), padding (0, 1)
        self.conv5 = nn.Conv2d(5, 5, kernel_size=(3, 8), stride=(2, 2), padding=(0, 1))  # 3x8 kernel, stride (2, 2), padding (0, 1)
        # self.conv6 = nn.Conv2d(5, 5, kernel_size=(4, 7), stride=(2, 1), padding=(0, 1))  # 4x7 kernel, stride (2, 1), padding (0, 1)

        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 2x2 pooling kernel, stride (2, 2)

        # Define the dropout layer
        self.dropout = nn.Dropout(0.5)

        # Define the fully connected layers
        self.fc1 = nn.Linear(30, 512)  # Adjust the input size based on the flattened output shape
        self.fc2 = nn.Linear(512, 128)
        self.fc_t = nn.Linear(128, 1)
        self.fc_a = nn.Linear(128, 1)
        self.fc_n = nn.Linear(128, 1)
        self.fc_s = nn.Linear(128, 1)

    def forward(self, x):
        # Apply the convolutional layers with ReLU activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # x = self.pool(F.relu(self.conv6(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Apply dropout
        x = self.dropout(x)

        # Apply the fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Apply the final fully connected layer
        t = torch.pow(self.fc_t(x),2.0)+0.01
        a = torch.pow(self.fc_a(x),2.0)+ 1e-16
        n_exponent = self.fc_n(x)
        n_samples = self.fc_s(x)
        return t, a, n_exponent, n_samples


if __name__ == "__main__":


    def read_audio(filename):
        waveform, samplerate = torchaudio.load(filename)
        return waveform.numpy(), samplerate  # Convert to numpy array immediately
        
    audio_file = f'/home/prsh7458/Desktop/scratch4/speech_data/motus_saved_noise_reverb_edc/reverberant_speech/reverberant_speech_{random.randint(0,200)}.wav'

    data, samplerate = read_audio(audio_file)



    feature_extractor = Log_spectral_feature_extraction(data,samplerate=samplerate)
    final_features = feature_extractor.main_process()
    

    cnn = R2Dnet()
    input_feature = torch.from_numpy(np.squeeze(final_features)).float()
    print(input_feature.shape)
    # sample_batch = torch.randn(5,1, 21, 511)
    t,a,n,s = cnn(input_feature.unsqueeze(0).unsqueeze(0))
    # output = cnn(sample_batch)
    print(t.shape)  


