import numpy as np
import tensorflow as tf
import keras
import os
import time
from datetime import datetime

'''
class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, data_path, ann_path, list_files, list_ann_files, 
                 batch_size=64, dim=(3000, 35), n_classes=6, shuffle=True):
        # Constructor of the data generator.
        self.dim = dim
        self.batch_size = batch_size
        self.data_path = data_path
        self.ann_path = ann_path
        self.list_files = list_files
        self.list_ann_files = list_ann_files
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.get_cnts() #Get the data count for each file        
        self.on_epoch_end() #Initialize file indexes        
        

    def __len__(self):
        # Denotes the number of batches per epoch
        return int((self.total_len+1) / self.batch_size)
    

    def __getitem__(self, index):
        
        start = index*self.batch_size
        end = min((index+1)*self.batch_size, self.total_len)
        
        X = np.empty((end - start,) + self.dim, dtype=np.float32)
        y = np.empty((end - start,), dtype=np.int32)
        
        curr_file_idx, accum_start, accum_end = self.get_accum_idx(index)
        
        curr_file = self.list_files[self.file_indexes[curr_file_idx]]
        curr_ann_file = self.list_ann_files[self.file_indexes[curr_file_idx]]
        data_index = self.data_indexes[self.file_indexes[curr_file_idx]]   

        curr_np = np.load(os.path.join(self.data_path, curr_file))
        curr_ann = np.load(os.path.join(self.ann_path, curr_ann_file))   
        curr_np = curr_np[data_index]
        curr_ann = curr_ann[data_index]  
        
        
        X_1 = curr_np[start - accum_start:end - accum_start] 
        y_1 = curr_ann[start - accum_start:end - accum_start]
        from_curr = min(accum_end - start, end - start)
        X[:from_curr] = np.expand_dims(X_1, axis=-1)
        y[:from_curr] = y_1
        
        if end > accum_end:
            curr_file_idx += 1
            accum_start = accum_end
            accum_end += self.list_cnt[self.file_indexes[curr_file_idx]]
            curr_file = self.list_files[self.file_indexes[curr_file_idx]]
            curr_ann_file = self.list_ann_files[self.file_indexes[curr_file_idx]]
            data_index = self.data_indexes[self.file_indexes[curr_file_idx]] 

            curr_np = np.load(os.path.join(self.data_path, curr_file))
            curr_ann = np.load(os.path.join(self.ann_path, curr_ann_file))
            curr_np = curr_np[data_index]
            curr_ann = curr_ann[data_index]
            
            X_2 = curr_np[:end - accum_start]
            y_2 = curr_ann[:end - accum_start]
            X[from_curr:] = np.expand_dims(X_2, axis=-1)
            y[from_curr:] = y_2
        #X = np.expand_dims(X, axis=-1)
        return X, y

    def get_accum_idx(self, index):
        curr_file_idx = 0
        accum_start = 0
        accum_end = self.list_cnt[self.file_indexes[0]]
        for i in range(len(self.file_indexes)):
            if index * self.batch_size < accum_end:
                curr_file_idx = i                
                break            
            accum_start += self.list_cnt[self.file_indexes[i]]
            accum_end += self.list_cnt[self.file_indexes[i+1]]
        
        return curr_file_idx, accum_start, accum_end
        

    def on_epoch_end(self):        
        self.curr_file_idx = 0
        # This function is called at the end of each epoch.
        self.file_indexes = np.arange(len(self.list_files)) #This is necessary to shuffle files
        self.data_indexes = [np.arange(cnt) for cnt in self.list_cnt]
        if self.shuffle == True:
            np.random.shuffle(self.file_indexes)
            for i in range(len(self.list_cnt)):
                np.random.shuffle(self.data_indexes[i]) 
        
            
    def get_cnts(self):
        list_cnt = []
        for f in self.list_files:
            temp_np = np.load(os.path.join(self.data_path, f))
            cnt_data = temp_np.shape[0]
            list_cnt.append(cnt_data)
            
        self.list_cnt = list_cnt
        self.total_len = sum(list_cnt)    
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_ratio', default=1.0, help='How much data to use for training, 1.0 to use entire data')
parser.add_argument('--include_type', default='SC', help='Use SC or ST for training')
parser.add_argument('--model', default=0, help='Model type. 0: Res34, 1:ConvAttn, 2:ConvAttn2, 3:ConvAttn3')
parser.add_argument('--num_epoch', default=30, help='Number of training epochs')
FLAGS = parser.parse_args()

np.random.seed(1)

dim_HT2D = (3000,35,1)
n_classes= 6
epochs = int(FLAGS.num_epoch)
bs = 64
BASE_LEARNING_RATE = 1e-3

import nets
from Data import datagen
import mobilenetV2_custom

model_flag = int(FLAGS.model)
if model_flag == 0:
    model = tf.keras.Sequential(
    [mobilenetV2_custom.MobileNetV2(
        input_shape=dim_HT2D,
        alpha=1.0,
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling='max'
    ),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(n_classes, activation='softmax')],
     name='mobilenet')

elif model_flag == 1:
    model = nets.Conv2DSimple()

x = np.random.random((1,) + dim_HT2D)
x = tf.convert_to_tensor(x)

print(model(x))
print(model.summary())


PROCESSED_DATA_PATH = os.path.join('/home','aiot','data','origin_npy')

save_signals_path_SC = os.path.join(PROCESSED_DATA_PATH,'HT2D_SC')
save_annotations_path_SC = os.path.join(PROCESSED_DATA_PATH,'annotations_SC')
save_signals_path_ST = os.path.join(PROCESSED_DATA_PATH,'HT2D_ST')
save_annotations_path_ST = os.path.join(PROCESSED_DATA_PATH,'annotations_ST')

def match_annotations_npy(dirname, filepath):
    filename = os.path.basename(filepath)
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filenames = [file for file in file_list if search_filename in file if file.endswith('.npy')]
    return filenames

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

    


'''
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

list_files_train = [f + '_HT2D.npy' for f in list_files_train]
list_files_test = [f + '_HT2D.npy' for f in list_files_test]

list_ann_files_train = []
list_ann_files_test = []
for f in list_files_train:
    ann_file = match_annotations_npy(save_annotations_path, f)
    list_ann_files_train.append(ann_file[0])
    
for f in list_files_test:
    ann_file = match_annotations_npy(save_annotations_path, f)
    list_ann_files_test.append(ann_file[0])
'''

train_generator = datagen.DataGenerator(list_files_train, list_ann_files_train, 
                          batch_size=bs, dim=dim_HT2D, n_classes=n_classes, shuffle=True)

test_generator = datagen.DataGenerator(list_files_test, list_ann_files_test, 
                          batch_size=bs, dim=dim_HT2D, n_classes=n_classes, shuffle=False)


LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

startTime = datetime.now()
log_fname = startTime.strftime("%Y%m%d %H%M")
LOG_FOUT = open(os.path.join(LOG_DIR, 'conv2d_' + model.name + '_' + log_fname + '.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for _ in range(epoch // 10):
        lr *= 0.1
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    optimizer.learning_rate = lr

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#loss_fn = weighted_categorical_crossentropy(weights=class_weight)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './ckpt' + model.name, max_to_keep=1)
start_epoch = 0
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    start_epoch = ckpt.step.numpy()-1
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
    print("")
    log_string("Training loss: %.6f, Training acc: %.3f"%(total_loss / total_cnt, correct / total_cnt))
    log_string("Training time: %.2f sec "%(time.time() - start))
    
    ckpt.step.assign_add(1)

    if e+1 >= 10 and (e+1) % 5 == 0:
        start = time.time()
        
        correct, total_cnt, total_loss = 0.0, 0.0, 0.0
        for idx, (x, y) in enumerate(test_generator):
            #y_pred = model(x, training=False)
            y_pred = test_step(x,y)
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
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))   


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


'''


def eval_model(test_gen, model):
    total_cnt = 0.0
    total_loss = 0.0
    correct = 0.0
    for x, y in test_gen:
        y_pred = model.predict(x)
        y_pred_cls = tf.math.argmax(y_pred, axis=-1)
        correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
        total_cnt += y_pred.shape[0]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                                reduction=tf.keras.losses.Reduction.SUM)
        y = tf.cast(y, dtype=tf.int32)
        total_loss += loss_fn(y, y_pred).numpy()
            
    test_acc = correct / total_cnt
    test_loss = total_loss / total_cnt
    
    print("test_acc: %.3f, test_loss: %.6f"%(test_acc, test_loss))

model.load_weights(os.path.join('ckpt_conv','ckpt_26'))
print("Checkpoint loaded")

callbacks = []
#Checkpoint설정
checkpoint_dir = './ckpt_conv2D'
model_cp_path = os.path.join(checkpoint_dir, "ckpt_{epoch}")
callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_cp_path, save_weights_only=True))


class eval_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        eval_model(test_generator, self.model)
callbacks.append(eval_callback())

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['sparse_categorical_accuracy'],
)

history = model.fit(train_generator,              
              epochs=epochs,
              verbose=1,
              callbacks=callbacks)

'''
'''
#Checkpoint설정
checkpoint_dir = './ckpt' + model.name
model_cp_path = os.path.join(checkpoint_dir, "ckpt_{epoch}")
callbacks = []
callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_cp_path, save_weights_only=True, save_best_only=True))

#Logging
class MyLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        curr_loss = logs.get('loss')
        curr_acc = logs.get('acc')
        test_loss = logs.get('val_loss')
        test_acc = logs.get('val_acc')        
        log_string('-'*20 + 'Epoch ' + str(epoch) + '-'*20)
        log_string("Training loss: %.6f, Training acc: %.3f"%(curr_loss, curr_acc))
        log_string("Test loss: %.6f, Test acc: %.3f"%(test_loss, test_acc))
        log_string("Training time(including test): %.2f sec "%(time.time() - self.starttime))
        self.starttime = time.time()
        

callbacks.append(MyLogger())

model.compile(optimizer, loss=loss_fn, metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_generator, validation_data=test_generator, epochs=epochs, callbacks=callbacks, use_multiprocessing=True, workers=8)
'''