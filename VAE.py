
import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
import keras.backend as K

import os
import time
from datetime import datetime



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_ratio', default=1.0, help='How much data to use for training, 1.0 to use entire data')
parser.add_argument('--include_type', default='SC', help='Use SC or ST for training')
parser.add_argument('--model', default=0, help='Model type. 0: Res34, 1:ConvAttn, 2:ConvAttn2, 3:ConvAttn3')
parser.add_argument('--num_epoch', default=30, help='Number of training epochs')
FLAGS = parser.parse_args()


ENVIRON = 'snuh'
np.random.seed(1)

dim_HT1D = (6000,1) if ENVIRON == 'snuh' else (3000,1)
n_classes = 5 if ENVIRON == 'snuh' else 6
epochs = int(FLAGS.num_epoch)
bs = 64
BASE_LEARNING_RATE = 1e-3

import nets
from Data import datagen
import importlib 
import resnet1D_Ahmed
importlib.reload(nets)  # Python 3.4+
model_flag = int(FLAGS.model)

model = VAE()


x = np.random.random((1,) + dim_HT1D)
x = tf.convert_to_tensor(x)
print(model(x))
print(model.name)

print(model.summary())

def match_annotations_npy(dirname, filepath):
    filename = os.path.basename(filepath)
    if ENVIRON == 'snuh':
        search_filename = filename.split('.')[0]
    else:
        search_filename = filename.split('-')[0][:-2]
        
    file_list = os.listdir(dirname)
    filenames = [f for f in file_list if search_filename in f if f.endswith('.npy')]
    return filenames

if ENVIRON == 'snuh':
    PROCESSED_DATA_PATH = os.path.join('/tf','00_data')
    save_signals_path = os.path.join(PROCESSED_DATA_PATH,'signals_filtered')
    save_annotations_path = os.path.join(PROCESSED_DATA_PATH,'sleep_edf','all_channels','annotations')
    list_files = [os.path.join(save_signals_path, f) for f in os.listdir(save_signals_path) if f.endswith('.npy')]
    train_test_split = 0.7
    split_cnt = int(train_test_split * len(list_files))

    list_files_train = []
    list_files_test = []

    list_ann_files_train = []
    list_ann_files_test = []

    list_files_train = np.random.choice(list_files[:split_cnt], int(float(FLAGS.data_ratio) * split_cnt), replace=False)
    list_files_train = list_files_train.tolist()
    for f in list_files_train:
        ann_file = match_annotations_npy(save_annotations_path, f)
        list_ann_files_train.append(os.path.join(save_annotations_path, ann_file[0]))

    list_files_test += list_files[split_cnt:]

    for f in list_files[split_cnt:]:
        ann_file = match_annotations_npy(save_annotations_path, f)
        list_ann_files_test.append(os.path.join(save_annotations_path, ann_file[0]))

else:
    PROCESSED_DATA_PATH = os.path.join('/home','aiot','data','origin_npy')
    save_signals_path_SC = os.path.join(PROCESSED_DATA_PATH,'signals_SC_filtered')
    save_annotations_path_SC = os.path.join(PROCESSED_DATA_PATH,'annotations_SC')
    save_signals_path_ST = os.path.join(PROCESSED_DATA_PATH,'signals_ST_filtered')
    save_annotations_path_ST = os.path.join(PROCESSED_DATA_PATH,'annotations_ST')

    list_files_SC = [os.path.join(save_signals_path_SC, f) for f in os.listdir(save_signals_path_SC) if f.endswith('.npy')]
    list_files_ST = [os.path.join(save_signals_path_ST, f) for f in os.listdir(save_signals_path_ST) if f.endswith('.npy')]

    train_test_split = 0.7
    split_cnt_SC = int(train_test_split * len(list_files_SC))
    split_cnt_ST = int(train_test_split * len(list_files_ST))

    SC_ST = [x for x in FLAGS.include_type.split(',')]
    include_SC = 'SC' in SC_ST
    include_ST = 'ST' in SC_ST

    list_files_train = []
    list_files_test = []

    list_ann_files_train = []
    list_ann_files_test = []

    if include_SC:
        list_files_SC_train = np.random.choice(list_files_SC[:split_cnt_SC], int(float(FLAGS.data_ratio) * split_cnt_SC), replace=False)
        list_files_train += list_files_SC_train.tolist()
        for f in list_files_SC_train:
            ann_file = match_annotations_npy(save_annotations_path_SC, f)
            list_ann_files_train.append(os.path.join(save_annotations_path_SC, ann_file[0]))

        list_files_test += list_files_SC[split_cnt_SC:]

        for f in list_files_SC[split_cnt_SC:]:
            ann_file = match_annotations_npy(save_annotations_path_SC, f)
            list_ann_files_test.append(os.path.join(save_annotations_path_SC, ann_file[0]))

    if include_ST:
        list_files_ST_train = np.random.choice(list_files_ST[:split_cnt_ST], int(float(FLAGS.data_ratio) * split_cnt_ST), replace=False)
        list_files_train += list_files_ST_train.tolist()
        for f in list_files_ST_train:
            ann_file = match_annotations_npy(save_annotations_path_ST, f)
            list_ann_files_train.append(os.path.join(save_annotations_path_ST, ann_file[0]))

        list_files_test += list_files_ST[split_cnt_ST:]
        for f in list_files_ST[split_cnt_ST:]:
            ann_file = match_annotations_npy(save_annotations_path_ST, f)
            list_ann_files_test.append(os.path.join(save_annotations_path_ST, ann_file[0]))

train_generator = datagen.DataGenerator(list_files_train, list_ann_files_train, 
                          batch_size=bs, dim=dim_HT1D, n_classes=n_classes, shuffle=True, balanced_sampling=False)
test_generator = datagen.DataGenerator(list_files_test, list_ann_files_test, 
                          batch_size=bs, dim=dim_HT1D, n_classes=n_classes, shuffle=False, balanced_sampling=False)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for _ in range(epoch // 10):
        lr *= 0.1
    lr = min(lr, 1e-6)
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    optimizer.learning_rate = lr

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

startTime = datetime.now()
log_fname = startTime.strftime("%Y%m%d %H%M")
LOG_FOUT = open(os.path.join(LOG_DIR, 'vae_' + model.name + '_' + log_fname + '.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def compute_loss(data, reconstruction, mu, log_var, alpha = 1):
    
    # Reconstruction loss-
    recon_loss = tf.reduce_mean(
        tf.reduce_sum(            
            tf.keras.losses.mean_squared_error(data, reconstruction),
            axis = (1, 2)
            )
        )
    
    # KL-divergence loss-    
    kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
    kl_loss = tf.reduce_mean(
        tf.reduce_sum(
            kl_loss,
            axis = 1
        )
    )

    total_loss = (recon_loss * alpha) + kl_loss
    
    return total_loss, recon_loss, kl_loss
    


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, '.tf_ckpt/ckpt_' + model.name, max_to_keep=1)
start_epoch = 0
# if manager.latest_checkpoint:
#    ckpt.restore(manager.latest_checkpoint)
#    start_epoch = ckpt.step.numpy()-1
best_test_acc = 0.0

@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        data_recon, mu, log_var = model(data, training=True)
        total_loss, recon_loss, kl_loss = compute_loss(data=data, reconstruction = data_recon, mu = mu, log_var = log_var)
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))    
    return total_loss, recon_loss, kl_loss

@tf.function
def test_step(data):
    data_recon, mu, log_var = model(data, training=True)
    total_loss, recon_loss, kl_loss = compute_loss(data=data, reconstruction = data_recon, mu = mu, log_var = log_var)
    return total_loss, recon_loss, kl_loss

for e in range(start_epoch, epochs):    
    total_cnt, total_loss, recon_loss, kl_loss = 0.0, 0.0, 0.0, 0.0
    log_string('-'*20 + 'Epoch ' + str(e) + '-'*20)
    adjust_learning_rate(optimizer, e)
    start = time.time()
    for idx, (x, y) in enumerate(train_generator):   
        train_total_loss, train_recon_loss, train_kl_loss = train_step(x)
        total_cnt += y.shape[0]        
        total_loss += train_total_loss
        recon_loss += train_recon_loss
        kl_loss += train_kl_loss
        if (idx + 1) % 10 == 0 or idx+1 == len(train_generator):
            print("[%d / %d] Training total loss: %.6f, recon_loss: %.6f, kl_loss: %.6f"%
                  (idx+1, len(train_generator), total_loss / total_cnt, recon_loss / total_cnt, kl_loss / total_cnt), end='\r', flush=True)        
        
    print("")
    log_string("Training total loss: %.6f, recon_loss: %.6f, kl_loss: %.6f"%
    (total_loss / total_cnt, recon_loss / total_cnt, kl_loss / total_cnt))
    log_string("Training time: %.2f sec "%(time.time() - start))
    
    
    #if (e+1 >= 10 and (e+1) % 5 == 0) or e == 0:
    if e % 3 == 0:
        start = time.time()
        
        total_cnt, total_loss, recon_loss, kl_loss = 0.0, 0.0, 0.0, 0.0
        for idx, (x, y) in enumerate(test_generator):            
            val_total_loss, val_recon_loss, val_kl_loss = test_step(x)            
            total_cnt += y.shape[0]
            total_loss += val_total_loss
            recon_loss += val_recon_loss
            kl_loss += val_kl_loss            
            if (idx + 1) % 10 == 0 or idx+1 == len(test_generator):
                print("[%d / %d] test total loss: %.6f, recon loss : %.6f, kl loss : %.6f"%
                    (idx+1, len(test_generator), total_loss / total_cnt, recon_loss / total_cnt, kl_loss / total_cnt),end='\r', flush=True)
            
        print("")
        log_string("test total loss: %.6f, recon loss : %.6f, kl loss : %.6f"%(total_loss / total_cnt, recon_loss / total_cnt, kl_loss / total_cnt))
        log_string("Eval time: %.2f sec"%(time.time() - start))
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = manager.save(checkpoint_number=ckpt.step)
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))   

    ckpt.step.assign_add(1)

LOG_FOUT.close()