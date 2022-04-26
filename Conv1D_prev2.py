
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


np.random.seed(1)

dim_HT1D = (3000,1)
n_classes=6
epochs = int(FLAGS.num_epoch)
bs = 64
PREV_CNT = 10
BASE_LEARNING_RATE = 1e-3

import nets
from Data import datagen
import importlib 
importlib.reload(nets)  # Python 3.4+
model_flag = int(FLAGS.model)
if model_flag == 0:    
    model = nets.Conv1DASPP_single()
    model2 = nets.Conv1DASPP_multi3(batch_size=bs, prev_cnt=PREV_CNT)


x = np.random.random((bs,3000,1))
x = tf.convert_to_tensor(x)
x2 = np.random.random((bs,PREV_CNT+1, 3000,1))
x2 = tf.convert_to_tensor(x2)
print(model(x))
print(model2(x2))
print(model2.name)
print(model2.summary())

PROCESSED_DATA_PATH = os.path.join('/home','aiot','data','origin_npy')
save_signals_path_SC = os.path.join(PROCESSED_DATA_PATH,'signals_SC_filtered')
save_annotations_path_SC = os.path.join(PROCESSED_DATA_PATH,'annotations_SC')
save_signals_path_ST = os.path.join(PROCESSED_DATA_PATH,'signals_ST_filtered')
save_annotations_path_ST = os.path.join(PROCESSED_DATA_PATH,'annotations_ST')

save_signals_path_SC_seq = os.path.join(PROCESSED_DATA_PATH,'signals_SC_seq')
save_annotations_path_SC_seq = os.path.join(PROCESSED_DATA_PATH,'annotations_SC_seq')
save_signals_path_ST_seq = os.path.join(PROCESSED_DATA_PATH,'signals_ST_seq')
save_annotations_path_ST_seq = os.path.join(PROCESSED_DATA_PATH,'annotations_ST_seq')

def match_annotations_npy(dirname, filepath):
    filename = os.path.basename(filepath)
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filenames = [file for file in file_list if search_filename in file if file.endswith('.npy')]
    return filenames


list_files_SC = [os.path.join(save_signals_path_SC, f) for f in os.listdir(save_signals_path_SC) if f.endswith('.npy')]
list_files_ST = [os.path.join(save_signals_path_ST, f) for f in os.listdir(save_signals_path_ST) if f.endswith('.npy')]

list_files_SC_seq = [os.path.join(save_signals_path_SC_seq, f) for f in os.listdir(save_signals_path_SC_seq) if f.endswith('.npy')]
list_files_ST_seq = [os.path.join(save_signals_path_ST_seq, f) for f in os.listdir(save_signals_path_ST_seq) if f.endswith('.npy')]

train_test_split = 0.7
split_cnt_SC = int(train_test_split * len(list_files_SC))
split_cnt_ST = int(train_test_split * len(list_files_ST))

SC_ST = [x for x in FLAGS.include_type.split(',')]
include_SC = 'SC' in SC_ST
include_ST = 'ST' in SC_ST

list_files_train = []
list_files_test = []

list_seq_files_train = []
list_seq_files_test = []

list_ann_files_train = []
list_ann_files_test = []

list_ann_seq_files_train = []
list_ann_seq_files_test = []

if include_SC:
    list_files_SC_train = np.random.choice(list_files_SC[:split_cnt_SC], int(float(FLAGS.data_ratio) * split_cnt_SC), replace=False)
    list_files_train += list_files_SC_train.tolist()

    for f in list_files_SC_train:
        ann_file = match_annotations_npy(save_annotations_path_SC, f)
        list_ann_files_train.append(os.path.join(save_annotations_path_SC, ann_file[0]))

        list_seq_files_train.append(os.path.join(save_signals_path_SC_seq, os.path.basename(f)))
        list_ann_seq_files_train.append(os.path.join(save_annotations_path_SC_seq, ann_file[0]))
    
    list_files_test += list_files_SC[split_cnt_SC:]

    for f in list_files_SC[split_cnt_SC:]:
        ann_file = match_annotations_npy(save_annotations_path_SC, f)
        list_ann_files_test.append(os.path.join(save_annotations_path_SC, ann_file[0]))
        list_seq_files_test.append(os.path.join(save_signals_path_SC_seq, os.path.basename(f)))
        list_ann_seq_files_test.append(os.path.join(save_annotations_path_SC_seq, ann_file[0]))


if include_ST:
    list_files_ST_train = np.random.choice(list_files_ST[:split_cnt_ST], int(float(FLAGS.data_ratio) * split_cnt_ST), replace=False)
    list_files_train += list_files_ST_train.tolist()
    for f in list_files_ST_train:
        ann_file = match_annotations_npy(save_annotations_path_ST, f)
        list_ann_files_train.append(os.path.join(save_annotations_path_ST, ann_file[0]))
        list_seq_files_train.append(os.path.join(save_signals_path_ST_seq, os.path.basename(f)))
        list_ann_seq_files_train.append(os.path.join(save_annotations_path_ST_seq, ann_file[0]))

    list_files_test += list_files_ST[split_cnt_ST:]
    for f in list_files_ST[split_cnt_ST:]:
        ann_file = match_annotations_npy(save_annotations_path_ST, f)
        list_ann_files_test.append(os.path.join(save_annotations_path_ST, ann_file[0]))
        list_seq_files_test.append(os.path.join(save_signals_path_ST_seq, os.path.basename(f)))
        list_ann_seq_files_test.append(os.path.join(save_annotations_path_ST_seq, ann_file[0]))

# Generator for training the model predicting from multi epoch
train_generator2 = datagen.DataGenerator(list_seq_files_train, list_ann_seq_files_train, 
                          batch_size=bs, dim=(PREV_CNT+1,) + dim_HT1D, n_classes=n_classes, shuffle=True)
test_generator2 = datagen.DataGenerator(list_seq_files_test, list_ann_seq_files_test, 
                          batch_size=bs, dim=(PREV_CNT+1,) + dim_HT1D, n_classes=n_classes, shuffle=False)
def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for _ in range(epoch // 10):
        lr *= 0.1
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    optimizer.learning_rate = lr

def get_focal_loss_sigmoid_on_multi_classification(y_true, y_pred, gamma=2):    
    #y_true = tf.squeeze(y_true)  # label example: [0,1,2,3]
    y_true = tf.cast(tf.one_hot(y_true, depth=n_classes), tf.float32)

    loss = -y_true * ((1 - y_pred) ** gamma) * K.log(y_pred + 1e-6)
    #loss = -y_true * K.log(y_pred + 1e-6)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss


LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

startTime = datetime.now()
log_fname = startTime.strftime("%Y%m%d %H%M")
LOG_FOUT = open(os.path.join(LOG_DIR, 'conv1d_' + model2.name + '_' + log_fname + '.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#loss_fn = weighted_categorical_crossentropy(weights=class_weight)
#loss_fn = get_focal_loss_sigmoid_on_multi_classification

log_string('='*20 + 'Training multi epoch model' + '='*20)

ckpt2 = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model2)
manager2 = tf.train.CheckpointManager(ckpt2, '.tf_ckpt/ckpt_' + model2.name, max_to_keep=1)


#Restore previously trained single epoch model
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, '.tf_ckpt/ckpt_' + model.name, max_to_keep=1)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    start_epoch = ckpt.step.numpy()-1

model2.aspp.set_weights(model.aspp.get_weights())

#if manager2.latest_checkpoint:
#    ckpt2.restore(manager2.latest_checkpoint)
#    start_epoch = ckpt2.step.numpy()-1

@tf.function
def train_step2(x, y):
    with tf.GradientTape() as tape:
        y_pred = model2(x, training=True)
        loss_value = loss_fn(y, y_pred)
    grads = tape.gradient(loss_value, model2.trainable_weights)
    optimizer.apply_gradients(zip(grads, model2.trainable_weights))    
    return loss_value, y_pred

@tf.function
def test_step2(x, y):
    y_pred = model2(x, training=False)
    return y_pred


start_epoch = 0
best_test_acc = 0.0

for e in range(start_epoch, epochs):
    correct, total_cnt, total_loss = 0.0, 0.0, 0.0
    log_string('-'*20 + 'Epoch ' + str(e) + '-'*20)
    adjust_learning_rate(optimizer, e)
    start = time.time()
    for idx, (x, y) in enumerate(train_generator2):   
        loss, y_pred = train_step2(x, y)

        total_cnt += y_pred.shape[0]
        y_pred_cls = tf.math.argmax(y_pred, axis=-1)
        correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
        total_loss += loss * y_pred.shape[0]
        if (idx + 1) % 10 == 0 or idx+1 == len(train_generator2):
            print("[%d / %d] Training loss: %.6f, Training acc: %.3f"%
                  (idx+1, len(train_generator2), total_loss / total_cnt, correct / total_cnt),end='\r', flush=True)
        
    print("")
    log_string("Training loss: %.6f, Training acc: %.3f"%(total_loss / total_cnt, correct / total_cnt))
    log_string("Training time: %.2f sec "%(time.time() - start))
    ckpt2.step.assign_add(1)
    
    if e==0 or (e+1 >= 10 and (e+1) % 5 == 0):
        start = time.time()
        
        correct, total_cnt, total_loss = 0.0, 0.0, 0.0
        for idx, (x, y) in enumerate(test_generator2):            
            y_pred = test_step2(x, y)
            y_pred_cls = tf.math.argmax(y_pred, axis=-1)
            correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
            total_cnt += y_pred.shape[0]

            y = tf.cast(y, dtype=tf.int32)            
            total_loss += loss_fn(y, y_pred).numpy() * y_pred.shape[0]            
                
            test_acc = correct / total_cnt
            test_loss = total_loss / total_cnt
            if (idx + 1) % 10 == 0 or idx+1 == len(test_generator2):
                print("[%d / %d] test loss: %.6f, test accuracy: %.3f"%
                    (idx+1, len(test_generator2), test_loss, test_acc),end='\r', flush=True)
            
        print("")
        log_string("test loss: %.6f, test acc: %.3f"%(test_loss, test_acc))
        log_string("Eval time: %.2f sec"%(time.time() - start))
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = manager2.save(checkpoint_number=ckpt.step)
            print("Saved checkpoint for step {}: {}".format(int(ckpt2.step), save_path))  

correct, total_cnt, total_loss = 0.0, 0.0, 0.0
confusion_matrix = np.zeros((n_classes,n_classes))
for idx, (x, y) in enumerate(test_generator2):
    y_pred = model2(x, training=False)
    y_pred_cls = tf.math.argmax(y_pred, axis=-1)

    y = tf.cast(y, dtype=tf.int32)    
    for i in range(n_classes):
        for j in range(n_classes):
            confusion_matrix[i,j] += np.sum((y_pred_cls[PREV_CNT:].numpy()==i) * (y[PREV_CNT:].numpy()==j))

log_string('-'*20 + 'Confusion Matrix' + '-'*20)
for i in range(n_classes):
    print_ln = ""
    for j in range(n_classes):
        print_ln += "%.3f "%(confusion_matrix[i,j] / np.sum(confusion_matrix[i]))
    log_string(print_ln)

log_string('-'*20 + 'Confusion Matrix Counts' + '-'*20)
for i in range(n_classes):
    print_ln = ""
    for j in range(n_classes):
        print_ln += "%d "%(confusion_matrix[i,j])
    log_string(print_ln)

LOG_FOUT.close()