
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
if model_flag == 0:
    model = resnet1D_Ahmed.eegnet()
elif model_flag == 1:
    model = nets.Conv1DAttention()
elif model_flag == 2:
    model = nets.Conv1DAttention2()
elif model_flag == 3:
    model = nets.Conv1DASPP()
elif model_flag == 4:
    model = nets.Conv1DASPP_1()
elif model_flag == 5:
    model = nets.Conv1DASPP_2()
elif model_flag == 6:
    model = nets.Conv1D_SPP()
elif model_flag == 7:
    dil_fac = dim_HT1D[0] // 3000
    model = nets.Conv1DASPP_single(dil_fac=dil_fac)

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
                          batch_size=bs, dim=dim_HT1D, n_classes=n_classes, shuffle=True, balanced_sampling=True)
test_generator = datagen.DataGenerator(list_files_test, list_ann_files_test, 
                          batch_size=bs, dim=dim_HT1D, n_classes=n_classes, shuffle=False, balanced_sampling=False)
# Calculate class weight
# Tested loss with class weight, but doesn't improve the accuracy
print(train_generator.list_cnt)

'''
from collections import defaultdict
cnt_class = defaultdict(int)
for x, y in train_generator:
    unique, counts = np.unique(y, return_counts=True)
    for i, cnt in zip(unique, counts):
        cnt_class[i] += cnt
cnt_class_np = np.zeros((n_classes,))
for i in range(n_classes):
    cnt_class_np[i] = cnt_class[i]
class_weight = 0.1 * np.ones((n_classes,))
class_weight[:n_classes-1] = np.sqrt(sum(cnt_class_np[:n_classes-1])/(n_classes * cnt_class_np[:n_classes-1]))
print("Count class", cnt_class_np)
print("Class weight", class_weight)
'''

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for _ in range(epoch // 10):
        lr *= 0.1
    lr = min(lr, 1e-6)
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    optimizer.learning_rate = lr

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        bs = y_pred.shape[0]
        y_onehot = tf.one_hot(y_true, depth=n_classes)
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_onehot * K.log(y_pred) * weights
        #loss = -K.sum(loss, -1)
        loss = -K.sum(loss) / bs
        return loss
    
    return loss


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
LOG_FOUT = open(os.path.join(LOG_DIR, 'conv1d_' + model.name + '_' + log_fname + '.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#loss_fn = weighted_categorical_crossentropy(weights=class_weight)
#loss_fn = get_focal_loss_sigmoid_on_multi_classification
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, '.tf_ckpt/ckpt_' + model.name, max_to_keep=1)
start_epoch = 0
#if manager.latest_checkpoint:
#    ckpt.restore(manager.latest_checkpoint)
#    start_epoch = ckpt.step.numpy()-1
best_test_acc = 0.0

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y, y_pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))    
    return loss_value, y_pred

@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    return y_pred

for e in range(start_epoch, epochs):
    correct, total_cnt, total_loss = 0.0, 0.0, 0.0
    log_string('-'*20 + 'Epoch ' + str(e) + '-'*20)
    adjust_learning_rate(optimizer, e)
    start = time.time()
    for idx, (x, y) in enumerate(train_generator):   
        '''
        y_onehot = tf.one_hot(y, depth=n_classes)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            #loss = loss_fn(y_onehot, y_pred)
            loss = loss_fn(y, y_pred)
        '''
        loss, y_pred = train_step(x, y)

        total_cnt += y_pred.shape[0]
        y_pred_cls = tf.math.argmax(y_pred, axis=-1)
        correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
        total_loss += loss * y_pred.shape[0]
        if (idx + 1) % 10 == 0 or idx+1 == len(train_generator):
            print("[%d / %d] Training loss: %.6f, Training acc: %.3f"%
                  (idx+1, len(train_generator), total_loss / total_cnt, correct / total_cnt),end='\r', flush=True)
        #grads = tape.gradient(loss, model.trainable_weights)
        #optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
    print("")
    log_string("Training loss: %.6f, Training acc: %.3f"%(total_loss / total_cnt, correct / total_cnt))
    log_string("Training time: %.2f sec "%(time.time() - start))
    
    
    #if (e+1 >= 10 and (e+1) % 5 == 0) or e == 0:
    if e % 3 == 0:
        start = time.time()
        
        correct, total_cnt, total_loss = 0.0, 0.0, 0.0
        for idx, (x, y) in enumerate(test_generator):
            #y_pred = model(x, training=False)
            y_pred = test_step(x, y)
            y_pred_cls = tf.math.argmax(y_pred, axis=-1)
            correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
            total_cnt += y_pred.shape[0]
            y = tf.cast(y, dtype=tf.int32)
            y_onehot = tf.one_hot(y, depth=n_classes)
            total_loss += loss_fn(y, y_pred).numpy() * y_pred.shape[0]
            #total_loss += loss_fn(y_onehot, y_pred).numpy() * y_pred.shape[0]
                
            test_acc = correct / total_cnt
            test_loss = total_loss / total_cnt
            if (idx + 1) % 10 == 0 or idx+1 == len(test_generator):
                print("[%d / %d] test loss: %.6f, test accuracy: %.3f"%
                    (idx+1, len(test_generator), test_loss, test_acc),end='\r', flush=True)
            
        print("")
        log_string("test loss: %.6f, test acc: %.3f"%(test_loss, test_acc))
        log_string("Eval time: %.2f sec"%(time.time() - start))
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = manager.save(checkpoint_number=ckpt.step)
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))   

    ckpt.step.assign_add(1)


correct, total_cnt, total_loss = 0.0, 0.0, 0.0
confusion_matrix = np.zeros((n_classes,n_classes))
for idx, (x, y) in enumerate(test_generator):
    y_pred = model(x, training=False)
    y_pred_cls = tf.math.argmax(y_pred, axis=-1)
    correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
    total_cnt += y_pred.shape[0]
    y = tf.cast(y, dtype=tf.int32)    
    for i in range(n_classes):
        for j in range(n_classes):
            confusion_matrix[i,j] += np.sum((y_pred_cls.numpy()==i) * (y.numpy()==j))


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