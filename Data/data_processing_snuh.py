import os, sys
import numpy as np
from tqdm import tqdm

data_folder = '/tf/00_data/sleep_edf/all_channels'
ann_folder = os.path.join(data_folder, 'annotations')

npy_files = os.listdir(data_folder)
ann_files = os.listdir(ann_folder)
npy_files = [f for f in npy_files if f.endswith('npy')]

sample_file = npy_files[1]
ann_file = sample_file.split('.')[0] + '_ann.npy'
sample_data = np.load(os.path.join(data_folder, sample_file)) #Shape: [23, 6000 * num_epoch]
sample_ann = np.load(os.path.join(ann_folder, ann_file)) #Shape: [num_epoch], data is one of ['N1', 'N2', 'N3', 'REM', 'Wake']


DATA_PATH = '/tf/00_data'
save_anns_path = os.path.join(DATA_PATH,'anns_number')
os.makedirs(save_anns_path, exist_ok=True)

EPOCH_SIZE = 30
FS = 200
SEQ_LENGTH = 10 # sequence of epochs length

'''
for ann in ann_files:    
    ann_np = np.load(os.path.join(ann_folder, ann))
    condlist = [ann_np == 'Wake', ann_np == 'N1', ann_np == 'N2', ann_np == 'N3', ann_np == 'REM']
    choicelist = [0, 1, 2, 3, 4]
    ann_number = np.select(condlist, choicelist, default=5)    
    np.save(os.path.join(save_anns_path, ann), ann_number)
''' 
    
''' Optimize with Numba, not very useful in this case
from numba import jit
import time

start = time.time()
cnt = 0

@jit(nopython=True)
def select_numba(np_data):
    out = np.zeros(np_data.shape[0])
    for i in range(np_data.shape[0]):
        if np_data[i] == 'Wake':
            out[i] = 0
        elif np_data[i] == 'N1':
            out[i] = 1
        elif np_data[i] == 'N2':
            out[i] = 2
        elif np_data[i] == 'N3':
            out[i] = 3
        elif np_data[i] == 'REM':
            out[i] = 4
        else:
            out[i] = 5
    return out

for ann in ann_files:    
    ann_np = np.load(os.path.join(ann_folder, ann))        
    ann_number = select_numba(ann_np)
    cnt += 1
    #if cnt > 1000:   
    #    break
end = time.time()
print(end - start)
'''

def search_signals_npy(dirname):
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith('.npy')]
    
    return filenames

def match_annotations_npy(dirname, filename):
    #search_filename = filename.split('-')[0][:-2]
    search_filename = filename.split('.')[0]
    file_list = os.listdir(dirname)
    filenames = [file for file in file_list if search_filename in file if file.endswith('.npy')]

    return filenames

# Filtering functions: Butterworth bandpass filter
# Notch filter(band stop filter)
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=8):
    #nyq = 0.5 * fs
    #low = lowcut / nyq
    #high = highcut / nyq
    low = lowcut
    high = highcut
    sos  = butter(order, [low, high], btype='band', fs=fs, output='sos', analog=False)
    return sos 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    sos  = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = lfilter(b, a, data)
    y = sosfilt(sos, data)
    return y

def notch_filter(data, f0, Q, fs):
    b, a = signal.iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y


save_signals_path = os.path.join(DATA_PATH,'signals_filtered_uniScale')
os.makedirs(save_signals_path, exist_ok=True)

def filter_signal(file_list, in_folder, output_folder):
    for signal_file in tqdm(file_list):
        data = np.load(os.path.join(in_folder, signal_file))
        fs = FS #Sample rate, in Hz
        seconds = EPOCH_SIZE
        num_samples = fs * seconds        
        
        single_channel = data[19] #use 19th channel ('C3-A2')
        data_reshape = single_channel.reshape((-1,num_samples)) #Shape: (num_epoch, 6000)        
        
        filtered_signal = []
        for i in range(len(data_reshape)):
            single_epoch = data_reshape[i]

            # Notch filter to cancel out the power line disturbance
            f0, Q = 50, 30
            y1 = notch_filter(single_epoch, f0, Q, fs)
            #print(y1)
            f0 = 60
            y2 = notch_filter(y1, f0, Q, fs)
            #print(y2)

            # Butterworth filter for valid freq signals
            low, high = 0.3, 35 
            filtered_epoch = butter_bandpass_filter(y2, low, high, fs, order=8)
            #print(filtered_epoch)
            
            ### NOrmalize...
            from sklearn.preprocessing import scale
            #filtered_epoch = scale(filtered_epoch)
            filtered_epoch = filtered_epoch/35/1e-6 # Uniform scaling by 35
            
            filtered_signal.append(filtered_epoch)

        filtered_signal = np.array(filtered_signal)

        np.save(os.path.join(output_folder, signal_file), filtered_signal)
        
        
filter_signal(npy_files, data_folder, save_signals_path)



save_seq_path = os.path.join(DATA_PATH,'signals_seq')
save_ann_seq_path = os.path.join(DATA_PATH,'annotations_seq')

os.makedirs(save_seq_path, exist_ok=True)
os.makedirs(save_ann_seq_path, exist_ok=True)

def convert_to_seq(file_list, in_folder, output_folder, ann_folder, ann_out_folder):
    
    for signal_file in tqdm(file_list):
        data = np.load(os.path.join(in_folder, signal_file)) #(# of epochs, fs * seconds)
        ann_file_name = match_annotations_npy(ann_folder, signal_file)
        print(ann_file_name)
        ann = np.load(os.path.join(ann_folder, ann_file_name[0]))
        
        num_epochs = ann.shape[0]
        seq_data = np.zeros((num_epochs - SEQ_LENGTH, SEQ_LENGTH + 1, EPOCH_SIZE * FS))
        seq_ann_data = np.zeros((num_epochs - SEQ_LENGTH,))
        
        for i in range(num_epochs - SEQ_LENGTH):
            #seq_data[i] = data[i:i+SEQ_LENGTH+1]
            seq_ann_data[i] = ann[i+SEQ_LENGTH]

        np.save(os.path.join(output_folder, signal_file), seq_data)
        np.save(os.path.join(ann_out_folder, ann_file_name[0]), seq_ann_data)
        
npy_signals = search_signals_npy(save_signals_path)
# convert_to_seq(npy_signals, save_signals_path, save_seq_path, ann_folder, save_ann_seq_path)