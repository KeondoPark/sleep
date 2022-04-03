import numpy as np
import pandas as pd
from pyedflib import highlevel
import matplotlib.pyplot as plt
import os
import random
import shutil
from  tqdm import tqdm


'''
Parameters
- 1 epoch = 30 seconds
- sample frequency = 100 Hz
'''
EPOCH_SIZE = 30
FS = 100
SEQ_LENGTH = 10 # sequence of epochs length

BASE_PATH = '/home/aiot/data/physionet.org/files/sleep-edfx/1.0.0/'
# EDF dataset path
src_path_ST = os.path.join(BASE_PATH, 'sleep-telemetry')
src_path_SC = os.path.join(BASE_PATH, 'sleep-cassette')

'''
File path
'''
data_path = os.path.join('/home','aiot','data')
PROCESSED_DATA_PATH = os.path.join(data_path,'origin_npy')
save_signals_path_SC = os.path.join(PROCESSED_DATA_PATH,'signals_SC')
save_annotations_path_SC = os.path.join(PROCESSED_DATA_PATH,'annotations_SC')
save_signals_path_ST = os.path.join(PROCESSED_DATA_PATH,'signals_ST')
save_annotations_path_ST = os.path.join(PROCESSED_DATA_PATH,'annotations_ST')
save_filtered_signals_path_ST = os.path.join(PROCESSED_DATA_PATH,'signals_ST_filtered')
save_filtered_signals_path_SC = os.path.join(PROCESSED_DATA_PATH,'signals_SC_filtered') 
save_seq_path_ST = os.path.join(PROCESSED_DATA_PATH,'signals_ST_seq')
save_ann_seq_path_ST = os.path.join(PROCESSED_DATA_PATH,'annotations_ST_seq')       
save_seq_path_SC = os.path.join(PROCESSED_DATA_PATH,'signals_SC_seq')
save_ann_seq_path_SC = os.path.join(PROCESSED_DATA_PATH,'annotations_SC_seq')   
            

'''
Make directory
*exist_ok = True : Make directory if there isn't directory in the path
'''
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(save_signals_path_SC, exist_ok=True)
os.makedirs(save_annotations_path_SC, exist_ok=True)
os.makedirs(save_signals_path_ST, exist_ok=True)
os.makedirs(save_annotations_path_ST, exist_ok=True)
os.makedirs(save_filtered_signals_path_ST, exist_ok=True)
os.makedirs(save_filtered_signals_path_SC, exist_ok=True)
os.makedirs(save_seq_path_ST, exist_ok=True)
os.makedirs(save_ann_seq_path_ST, exist_ok=True)
os.makedirs(save_seq_path_SC, exist_ok=True)
os.makedirs(save_ann_seq_path_SC, exist_ok=True)

# Filtering functions: Butterworth bandpass filter
# Notch filter(band stop filter)
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=8):
    #nyq = 0.5 * fs
    #low = lowcut / nyq
    #high = highcut / nyq
    #b, a = butter(order, [low, high], btype='band')
    low = lowcut
    high = highcut
    sos  = butter(order, [low, high], btype='band', fs=fs, output='sos', analog=False)
    return sos 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    sos  = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = lfilter(b, a, data)
    y = sosfilt(sos, data)
    return y

def butter_bandpass_orig(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    return b, a

def butter_bandpass_filter_orig(data, lowcut, highcut, fs, order=8):
    b,a  = butter_bandpass_orig(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)    
    return y

def notch_filter(data, f0, Q, fs):
    b, a = signal.iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, f0, Q, fs):
    b, a = signal.iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y

def search_signals_edf(dirname): 
    '''
    find signals files (PSG.edf)
    * Input
    - dirname: file path
    '''
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith('PSG.edf')]
    
    return filenames

def search_annotations_edf(dirname): 
    '''
    find annotations files (Hypnogram.edf)
    * Input
    - dirname: file path
    '''
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith('Hypnogram.edf')]
    
    return filenames

def match_annotations(dirname, filename): 
    '''
    find corresponding annotation files
    * Input
    - dirname: file path
    - filename: PSG.edf file name
    '''
    match_filename = filename.split('-')[0][:-2]
    # ex> 'ST7062J0-PSG.edf' -> ST7062
    file_list = os.listdir(dirname)
    filename = [file for file in file_list if match_filename in file if file.endswith("Hypnogram.edf")]

    return filename

def preprocess_signal(src_path, edf_list, output_signal_path, output_ann_path, epoch_size=30, sample_rate=100):

    for filename in edf_list:
        
        # Signal file path & Annotation file path
        signal_path = os.path.join(src_path, filename)
        annotation_path = os.path.join(src_path, match_annotations(src_path,filename)[0])

        
        # Read signal
        signals, signal_headers, header = highlevel.read_edf(signal_path)
        signal_len = len(signals[0]) // sample_rate // epoch_size 
        
        # Read annotation: find labels for epochs
        try:
            _, _, annotations = highlevel.read_edf(annotation_path)
        except Exception as e:
            print(str(e))
            continue

        label = []
        for ann in annotations['annotations']:
            '''
            annotations['annotations'] = [[start_time, duration, event_label], ...]
            W(0): Sleep stage W
            N1(1): Sleep stage 1
            N2(2): Sleep stage 2
            N3(3): Sleep stage 3 & 4
            REM(4): Sleep stage R
            Unknown(5)
            '''
            start_time = ann[0]
            
            duration = ann[1] # duration in seconds
            duration = int(duration) // epoch_size # duration in epoch
            
            event_label = ann[2]

            if event_label == 'Sleep stage W':
                for event in range(duration):
                    label.append(0)
            elif event_label == 'Sleep stage 1':
                for event in range(duration):
                    label.append(1)
            elif event_label == 'Sleep stage 2':
                for event in range(duration):
                    label.append(2)
            elif event_label == 'Sleep stage 3' or event_label == 'Sleep stage 4':
                for event in range(duration):
                    label.append(3)
            elif event_label == 'Sleep stage R':
                for event in range(duration):
                    label.append(4)
            else:
                for event in range(duration):
                    label.append(5)
                    
        label = np.array(label)
        annotations_len = len(label)
        
        
        # Cut off the signal data at the length of the sleep stage labels (annotation)
        # Select 'Fpz-Cz' only (singals[0])
        if header['startdate'] == annotations['startdate']:
            print("%s file's signal & annotations start time is same"%signal_path.split('/')[-1])

            if signal_len > annotations_len :
                signals = signals[0][:(epoch_size * sample_rate * annotations_len)]
            else: # signal_len < annotations_len
                signals = signals[0][:(epoch_size * sample_rate * signal_len)]
                label = label[:signal_len]

            signals = np.array(signals)
            signals = signals.reshape(-1,epoch_size * sample_rate)
            
            # Save npy file
            np.save(os.path.join(output_signal_path, signal_path.split('/')[-1].split('.')[0]),signals)
            np.save(os.path.join(output_ann_path, annotation_path.split('/')[-1].split('.')[0]),label)
            #np.save(os.path.join(save_annotations_path2, annotation_path.split('/')[-1].split('.')[0]),label[1:-1])

            #if (len(signals) // (epoch_size * sample_rate) != len(label)):
            if (len(signals) != len(label)):
                print('signals len : %d / annotations len : %d'%(len(signals),len(label)))
                #print('signals len : %d / annotations len : %d'%(len(signals) // (epoch_size * sample_rate),len(label)))

        else:
            print("%s file''s signal & annotations start time is different"%signal_path.split('/')[-1])

def search_signals_npy(dirname):
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith('.npy')]
    
    return filenames

def match_annotations_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filenames = [file for file in file_list if search_filename in file if file.endswith('.npy')]

    return filenames

def filter_signal(file_list, in_folder, output_folder):
    for signal_file in tqdm(file_list):
        data = np.load(os.path.join(in_folder, signal_file))
        fs = 100 #Sample rate, in Hz
        seconds = 30
        num_samples = fs * seconds

        # Filter out too small or large amplitude
        #data_limit = np.minimum(np.maximum(data, -150), 150)

        #data_reshape = data.reshape((-1,num_sample))
        filtered_signal = []
        for i in range(len(data)):
            single_epoch = data[i]

            # Notch filter to cancel out the power line disturbance
            f0, Q = 50, 30
            y1 = notch_filter(single_epoch, f0, Q, fs)
            #f0 = 60
            #y2 = notch_filter(y1, f0, Q, fs)

            # Butterworth filter for valid freq signals
            low, high = 0.3, 35 
            filtered_epoch = butter_bandpass_filter(y1, low, high, fs)
            
            ### NOrmalize...
            from sklearn.preprocessing import scale
            filtered_epoch = scale(filtered_epoch)
            
            filtered_signal.append(filtered_epoch)

        filtered_signal = np.array(filtered_signal)

        np.save(os.path.join(output_folder, signal_file), filtered_signal)

def convert_to_seq(file_list, in_folder, output_folder, ann_folder, ann_out_folder):
    
    for signal_file in tqdm(file_list):
        data = np.load(os.path.join(in_folder, signal_file)) #(# of epochs, fs * seconds)
        ann_file_name = match_annotations_npy(ann_folder, signal_file)
        ann = np.load(os.path.join(ann_folder, ann_file_name[0]))
        
        num_epochs = ann.shape[0]
        seq_data = np.zeros((num_epochs - SEQ_LENGTH, SEQ_LENGTH + 1, EPOCH_SIZE * FS))
        seq_ann_data = np.zeros((num_epochs - SEQ_LENGTH,))
        
        for i in range(num_epochs - SEQ_LENGTH):
            #seq_data[i] = data[i:i+SEQ_LENGTH+1]
            seq_ann_data[i] = ann[i+SEQ_LENGTH]

        np.save(os.path.join(output_folder, signal_file), seq_data)
        np.save(os.path.join(ann_out_folder, ann_file_name[0]), seq_ann_data)


from scipy import signal, ndimage
import emd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def ht_transform(data, visualize=False):
    '''
    data = signals.reshape(3000)[e]
    e: e th epoch
    i: i th file in npy_signals
    '''
    fs = 100 #Sample rate, in Hz
    seconds = 30
    num_samples = fs * seconds
    
    # Notch filter to cancel out the power line disturbance
    f0, Q = 50, 30
    y1 = notch_filter(data, f0, Q, fs)
    #f0 = 60
    #y2 = notch_filter(y1, f0, Q, fs)

    # Butterworth filter for valid freq signals
    low, high = 0.3, 35 
    filtered_epoch = butter_bandpass_filter(y1, low, high, fs)
    
    # Estimate IMFs    
    #imf = imf[fs * seconds:-fs * seconds]
    imf = emd.sift.sift(filtered_epoch)

    # Compute instantaneous phase, frequency and amplitude using the Normalised Hilbert Transform Method
    IP, IF, IA = emd.spectra.frequency_transform(imf, fs, 'hilbert')
    
    #IA = np.log(IA+1)

    # Compute Hilbert-Huang spectrum
    freq_range = (0.3, 35, 35)    
    """
    imf = emd.sift.sift(filtered_signal, max_imfs=5)
    for i in range(5):
        hht_f, hht = emd.spectra.hilberthuang(IF[:,i,None], IA[:,i,None], freq_range, mode='amplitude', sum_time=False)
        if i == 0:
            hht_stack = np.expand_dims(hht, -1)
        else:
            hht_stack = np.concatenate((hht_stack, np.expand_dims(hht, -1)), -1)
    """     
    
    
    hht_f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, mode='amplitude', sum_time=False)
    # Gaussian filter: 각 픽셀을 중심으로 주변 Pixel들을 Weighted average하여 픽셀 값 계산.
    #                  이때 Weight는 Gaussian 분포를 이루도록 설정
    #                  Noise 제거, Blur 효과를 주기 위해 사용
    #hht = ndimage.gaussian_filter(hht, 1)        
    
    ### NOrmalize...
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    hht_scaled = scaler.fit_transform(hht.reshape(35*3000,1))
    hht_scaled = hht_scaled.reshape(35,3000)

    if visualize:
        time_vect = np.linspace(0, hht.shape[-1] / fs, hht.shape[-1])
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(time_vect, hht_f, hht, cmap='viridis')
        plt.ylim(0, 35)
        plt.title('Hilbert-Huang Transform')
        plt.xlim(1, hht.shape[-1] / fs)
        plt.show()
    
    return hht_scaled, hht_f, IP, IF, IA


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--preprocess_edf', action='store_true', help='Preprocess edf files into npy files')
parser.add_argument('--filter', action='store_true', help='Filter signals - notch and butterworth')
parser.add_argument('--ht', action='store_true', help='Do Hilbert-Huang transform')
parser.add_argument('--include_type', default='SC', help='Processing for SC or ST')
parser.add_argument('--make_seq', action='store_true', help='Make sequence of epochs(e.g. 10 epochs) as one record')
FLAGS = parser.parse_args()


if __name__ == '__main__':
    SC_ST = [x for x in FLAGS.include_type.split(',')]
    include_SC = 'SC' in SC_ST
    include_ST = 'ST' in SC_ST
    if not include_SC and not include_ST:
        print("Either ST or SC should be included in 'include_type' argument")
        exit(-1)

    if FLAGS.preprocess_edf:
        if include_ST:
            #Extract signals EDF files
            signals_edf_list_ST = search_signals_edf(src_path_ST)        

            # Do preprocessing
            preprocess_signal(src_path=src_path_ST, edf_list=signals_edf_list_ST, \
                output_signal_path=save_signals_path_ST, output_ann_path=save_annotations_path_ST, epoch_size=EPOCH_SIZE, sample_rate=FS)

        if include_SC:
            signals_edf_list_SC = search_signals_edf(src_path_SC)
            preprocess_signal(src_path=src_path_SC, edf_list=signals_edf_list_SC, \
                output_signal_path=save_signals_path_SC, output_ann_path=save_annotations_path_SC, epoch_size=EPOCH_SIZE, sample_rate=FS)

    if FLAGS.filter:  
        if include_ST:      
            npy_signals_ST = search_signals_npy(save_signals_path_ST)            
            # Do filtering
            filter_signal(npy_signals_ST, save_signals_path_ST, save_filtered_signals_path_ST)

        if include_SC: 
            npy_signals_SC = search_signals_npy(save_signals_path_SC)            

            # Do filtering
            filter_signal(npy_signals_SC, save_signals_path_SC, save_filtered_signals_path_SC)

    if FLAGS.make_seq:
        if include_ST:                  
            npy_signals_ST = search_signals_npy(save_filtered_signals_path_ST)
            convert_to_seq(npy_signals_ST, save_filtered_signals_path_ST, save_seq_path_ST, save_annotations_path_ST, save_ann_seq_path_ST)

        if include_SC:                  
            npy_signals_SC = search_signals_npy(save_filtered_signals_path_SC)            
            convert_to_seq(npy_signals_SC, save_filtered_signals_path_SC, save_seq_path_SC, save_annotations_path_SC, save_ann_seq_path_SC)

    if FLAGS.ht:
        if include_SC:   
            HT2D_SC_path = os.path.join(PROCESSED_DATA_PATH, 'HT2D_SC')
            os.makedirs(HT2D_SC_path, exist_ok=True)

            """
            def read_csv_to_list(filepath):
                import csv
                with open(filepath, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    list_filepath = [row[0] for row in spamreader]
                return list_filepath

            SC_train = os.path.join('/home','aiot','data','origin_npy','SC_train.csv')
            SC_test = os.path.join('/home','aiot','data','origin_npy','SC_test.csv')

            list_files_train = read_csv_to_list(SC_train)
            list_files_test = read_csv_to_list(SC_test)

            list_files_train = [f + '.npy' for f in list_files_train]
            list_files_test = [f + '.npy' for f in list_files_test]

            npy_signals = list_files_train + list_files_test
            """
            npy_signals = search_signals_npy(save_signals_path_SC)

            for signal_file in tqdm(npy_signals):
                signal_data = np.load(os.path.join(save_signals_path_SC, signal_file))
                ht_signal_arr = []
                for signal_epoch in signal_data:
                    ht_signal_data, ht_signal_data_f, IP, IF, IA = ht_transform(signal_epoch)     
                    ht_signal_data = ht_signal_data.transpose()
                    ht_signal_arr.append(ht_signal_data)
                ht_signal_arr = np.array(ht_signal_arr)
                np.save(os.path.join(HT2D_SC_path, (signal_file.split('.')[0] + '_HT2D')), ht_signal_arr)

        if include_ST:  

            HT2D_ST_path = os.path.join(PROCESSED_DATA_PATH, 'HT2D_ST')
            os.makedirs(HT2D_ST_path, exist_ok=True)

            npy_signals_ST = search_signals_npy(save_signals_path_ST)            

            for signal_file in tqdm(npy_signals_ST):
                signal_data = np.load(os.path.join(save_signals_path_ST, signal_file))
                ht_signal_arr = []
                for signal_epoch in signal_data:
                    ht_signal_data, ht_signal_data_f, IP, IF, IA = ht_transform(signal_epoch)     
                    ht_signal_data = ht_signal_data.transpose()
                    ht_signal_arr.append(ht_signal_data)
                ht_signal_arr = np.array(ht_signal_arr)
                np.save(os.path.join(HT2D_ST_path, (signal_file.split('.')[0] + '_HT2D')), ht_signal_arr)


        

    