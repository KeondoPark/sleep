
import numpy as np
import tensorflow as tf
import keras
import os
from tqdm import tqdm
import time
import keras.backend as K
import numpy as np
import tensorflow as tf
import keras
import os
from tqdm import tqdm
import time
import keras.backend as K

np.random.seed(1)
class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, data_path, ann_path, list_files, list_ann_files, 
                 batch_size=64, dim=(3000,1), n_classes=5, shuffle=True):
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
            data_index = self.data_indexes[self.file_indexes[curr_file_idx]]
            
            
            curr_ann_file = self.list_ann_files[self.file_indexes[curr_file_idx]]
            curr_np = np.load(os.path.join(self.data_path, curr_file))
            curr_ann = np.load(os.path.join(self.ann_path, curr_ann_file))
            curr_np = curr_np[data_index]
            curr_ann = curr_ann[data_index]
            #curr_np = curr_np.reshape(-1, 3000, 1)
            
            #curr_np = curr_np[1:-1]
            #curr_ann = curr_ann[1:-1]
            
            X_2 = curr_np[:end - accum_start]
            y_2 = curr_ann[:end - accum_start]
            X[from_curr:] = np.expand_dims(X_2, axis=-1)
            y[from_curr:] = y_2        
      
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
            
        #self.accum_start = 0 
        #self.accum_end = self.list_cnt[self.file_indexes[0]]                 
            
    def get_cnts(self):
        list_cnt = []
        for f in self.list_files:
            temp_np = np.load(os.path.join(self.data_path, f))
            cnt_data = temp_np.shape[0] 
            list_cnt.append(cnt_data)
            
        self.list_cnt = list_cnt
        self.total_len = sum(list_cnt)    
#curr_path = os.getcwd() + '/'
PROCESSED_DATA_PATH = os.path.join('/home','aiot','data','origin_npy')
save_signals_path = os.path.join(PROCESSED_DATA_PATH,'signals_SC_filtered')
save_annotations_path = os.path.join(PROCESSED_DATA_PATH,'annotations_SC')
def match_annotations_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filenames = [file for file in file_list if search_filename in file if file.endswith('.npy')]
    return filenames
dim_HT1D = (3000,1)
n_classes=6
epochs = 50
bs = 64
BASE_LEARNING_RATE = 1e-3
list_files = [f for f in os.listdir(save_signals_path) if f.endswith('.npy')]
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
list_ann_files_train = []
list_ann_files_test = []
for f in list_files_train:
    ann_file = match_annotations_npy(save_annotations_path, f)
    list_ann_files_train.append(ann_file[0])
    
for f in list_files_test:
    ann_file = match_annotations_npy(save_annotations_path, f)
    list_ann_files_test.append(ann_file[0])
train_generator = DataGenerator(save_signals_path, save_annotations_path, list_files_train, list_ann_files_train, 
                          batch_size=bs, dim=dim_HT1D, n_classes=n_classes, shuffle=True)
test_generator = DataGenerator(save_signals_path, save_annotations_path, list_files_test, list_ann_files_test, 
                          batch_size=bs, dim=dim_HT1D, n_classes=n_classes, shuffle=False)
# Calculate class weight
# Tested loss with class weight, but doesn't improve the accuracy
​from collections import defaultdict
cnt_class = defaultdict(int)
for x, y in train_generator:
    unique, counts = np.unique(y, return_counts=True)
    for i, cnt in zip(unique, counts):
        cnt_class[i] += cnt
cnt_class_np = np.array(list(cnt_class.values()))
class_weight = sum(cnt_class_np)/(n_classes * cnt_class_np)

import nets
import importlib 
importlib.reload(nets)  # Python 3.4+
model = nets.Conv1DAttention2()
x = np.random.random((1,3000,1))
x = tf.convert_to_tensor(x)
print(model(x))

print(model.summary())

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for _ in range(epoch // 10):
        lr *= 0.1
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
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        #loss = -K.sum(loss, -1)
        loss = -K.sum(loss) / bs
        return loss
    
    return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss_fn = weighted_categorical_crossentropy(weights=class_weight)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './ckpt_Advanced_Conv1D', max_to_keep=1)
start_epoch = 0
#if manager.latest_checkpoint:
#    ckpt.restore(manager.latest_checkpoint)
#    start_epoch = ckpt.step.numpy()-1
best_test_acc = 0.0
for e in range(start_epoch, epochs):
    correct, total_cnt, total_loss = 0.0, 0.0, 0.0
    print('-'*20, 'Epoch ' + str(e) + '-'*20)
    adjust_learning_rate(optimizer, e)
    start = time.time()
    for idx, (x, y) in enumerate(train_generator):   
        y_onehot = tf.one_hot(y, depth=n_classes)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y_onehot, y_pred)
            #loss = loss_fn(y, y_pred)
        
        total_cnt += y_pred.shape[0]
        y_pred_cls = tf.math.argmax(y_pred, axis=-1)
        correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
        total_loss += loss * y_pred.shape[0]
        if (idx + 1) % 10 == 0 or idx+1 == len(train_generator):
            print("[%d / %d] Training loss: %.6f, Training acc: %.3f"%
                  (idx+1, len(train_generator), total_loss / total_cnt, correct / total_cnt),end='\r', flush=True)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print("")
    print("Training time: %.2f sec "%(time.time() - start))
    
    start = time.time()
    
    correct, total_cnt, total_loss = 0.0, 0.0, 0.0
    for idx, (x, y) in enumerate(test_generator):
        y_pred = model(x, training=False)
        y_pred_cls = tf.math.argmax(y_pred, axis=-1)
        correct += tf.reduce_sum(tf.cast(tf.equal(y_pred_cls, y), tf.float32))
        total_cnt += y_pred.shape[0]
        y = tf.cast(y, dtype=tf.int32)
        y_onehot = tf.one_hot(y, depth=n_classes)
        #total_loss += loss_fn(y, y_pred).numpy() * y_pred.shape[0]
        total_loss += loss_fn(y_onehot, y_pred).numpy() * y_pred.shape[0]
            
        test_acc = correct / total_cnt
        test_loss = total_loss / total_cnt
        if (idx + 1) % 10 == 0 or idx+1 == len(test_generator):
            print("[%d / %d] test loss: %.6f, test accuracy: %.3f"%
                  (idx+1, len(test_generator), test_loss, test_acc),end='\r', flush=True)
    print("")
    print("Eval time: %.2f sec"%(time.time() - start))
    ckpt.step.assign_add(1)
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
