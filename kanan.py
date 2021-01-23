import pyaudio
import struct
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
from scipy import signal
from scipy.fftpack import fft
import time
import pandas as pd
plt.style.use('dark_background')
import os
import random


# constants
p = pyaudio.PyAudio()

CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
#RATE = 44100                 # samples per second
RATE = 5000


# create matplotlib figure and axes
fig, (ax1,ax2) = plt.subplots(2, figsize=(5, 10))

CHUNK = int(RATE/20)
# stream object to get data from microphone
stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,output=True,frames_per_buffer=CHUNK)
# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)       # samples (waveform)
xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)
# create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=1)
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=1)

# format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(0, 255)
ax1.set_xlim(0, 2 * CHUNK)

# format spectrum axes
ax2.set_xlim(20, RATE / 2)

plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])
print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

ACTION = 'kanan' 
channel_datas = []

#while True:
for i in range (10):
    # binary data
    data = stream.read(CHUNK)  
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128
    line.set_ydata(data_np)
    # compute FFT and update line
    yf = fft(data_int)

    line_fft.set_ydata(np.abs(yf[0:CHUNK])  / (128 * CHUNK))
    data =np.abs(yf[0:CHUNK])  / (128 * CHUNK)

    network_input = np.array(data).reshape((-1,250))
    print(network_input.shape)
    channel_datas.append(data)
    
   
    plt.pause(0.005)

datadir = "data_baru"
if not os.path.exists(datadir):
    os.mkdir(datadir)

actiondir = f"{datadir}/{ACTION}"
if not os.path.exists(actiondir):

    os.mkdir(actiondir)
print(len(channel_datas))
print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))

  