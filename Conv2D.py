import numpy as np
import tensorflow as tf
import keras
import os


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

#curr_path = os.getcwd() + '/'
PROCESSED_DATA_PATH = os.path.join('/home','aiot','data','origin_npy')
HT2D_path = os.path.join(PROCESSED_DATA_PATH, 'HT2D_SC')
save_signals_path = os.path.join(PROCESSED_DATA_PATH,'signals_SC')
save_annotations_path = os.path.join(PROCESSED_DATA_PATH,'annotations_SC')


def match_annotations_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filenames = [file for file in file_list if search_filename in file if file.endswith('.npy')]

    return filenames

dim_HT2D = (3000,35,1)
n_classes=6
epochs = 50
bs = 64
list_files = [f for f in os.listdir(HT2D_path) if f.endswith('.npy')]

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


train_generator = DataGenerator(HT2D_path, save_annotations_path, list_files_train, list_ann_files_train, 
                          batch_size=bs, dim=dim_HT2D, n_classes=n_classes, shuffle=True)

test_generator = DataGenerator(HT2D_path, save_annotations_path, list_files_test, list_ann_files_test, 
                          batch_size=bs, dim=dim_HT2D, n_classes=n_classes, shuffle=False)



#import nets
#model = nets.Conv2DSimple()
import mobilenetV2_custom

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
     tf.keras.layers.Dense(n_classes, activation='softmax')])

x = np.random.random((1,) + dim_HT2D)
x = tf.convert_to_tensor(x)

print(model(x))
print(model.summary())


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