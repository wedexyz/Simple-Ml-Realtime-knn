import pyaudio
import struct
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm

import numpy as np
from scipy import signal
from scipy.fftpack import fft
import time

#plt.style.use('dark_background')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.multiclass import OneVsRestClassifier
import pandas as  pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
#import random

# constants
p = pyaudio.PyAudio()

CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
#RATE = 44100                 # samples per second
RATE = 5000


# create matplotlib figure and axes
fig, (ax1,ax2) = plt.subplots(2, figsize=(5, 5))

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


ACTIONS = ["idle","kiri","kanan"]

def create_data(starting_dir="data_baru"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []
        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])  
        training_data[action] = training_data[action][:min(lengths)]

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:
            if action == "idle":
                combined_data.append([data, [1, 0,0]])
            elif action == "kiri":
                combined_data.append([data, [0, 1,0]])
            elif action == "kanan":
                combined_data.append([data, [0, 0,1]])
    np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data

print("creating training data")
traindata = create_data(starting_dir="data_baru")
train_X = []
train_y = []

for X, y in traindata:
    train_X.append(X)
    train_y.append(y)

train_X = np.array(train_X).reshape(-1,250)
train_y = np.array(train_y)

x_train,x_test,y_train,y_test=train_test_split(train_X  ,train_y ,test_size=0.2)

y_tn=np.argmax(y_train, axis=1)
y_tt=np.argmax(y_test, axis=1)

model= KNeighborsClassifier()
#model = RandomForestClassifier()
#clf =OneVsRestClassifier(kn)
model.fit(x_train,y_tn)
predictions = model.predict(x_test)
print(classification_report(y_tt,predictions))


while True:
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

    network_input = np.array(data).reshape(-1,250)
    predictions= model.predict(network_input)
    #print(predictions[0])
    a = predictions[0]
    if a == 0 :
        print("idle")
    elif a >1 :
        print("kanan")
    else :
        print("kiri")

    #print(network_input)
    plt.pause(0.005)

  