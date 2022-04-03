import os, sys
import tensorflow as tf
import numpy as np
from collections import defaultdict

np.random.seed(1)

class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_files, list_ann_files, 
                 batch_size=64, dim=(3000,1), n_classes=5, shuffle=True, balanced_sampling=False):
        # Constructor of the data generator.
        self.dim = dim
        self.batch_size = batch_size
        self.list_files = list_files
        self.list_ann_files = list_ann_files
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.balanced_sampling = balanced_sampling # Balanced sampling across different classes(Excpet unknown(5) class)        
        self.on_epoch_end() #Initialize file indexes        
        
    def __len__(self): 
        # Denotes the number of batches per epoch
        return int((self.total_len+self.batch_size+1) / self.batch_size)
    
    def __getitem__(self, index):
        
        start = index*self.batch_size
        end = min((index+1)*self.batch_size, self.total_len)
        
        X = np.empty((end - start,) + self.dim, dtype=np.float32)
        y = np.empty((end - start,), dtype=np.int32)
        
        curr_file_idx, accum_start, accum_end = self.get_accum_idx(index)
        
        curr_file = self.list_files[self.file_indexes[curr_file_idx]]
        curr_ann_file = self.list_ann_files[self.file_indexes[curr_file_idx]]
        data_index = self.data_indexes[self.file_indexes[curr_file_idx]]
        
        curr_np = np.load(curr_file)
        curr_ann = np.load(curr_ann_file)
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
            curr_np = np.load(curr_file)
            curr_ann = np.load(curr_ann_file)
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
        self.get_cnts() #Get the data count for each file        
        self.curr_file_idx = 0
        # This function is called at the end of each epoch.
        self.file_indexes = np.arange(len(self.list_files)) #This is necessary to shuffle files        
        if self.shuffle == True:
            np.random.shuffle(self.file_indexes)
            for i in range(len(self.list_cnt)):
                np.random.shuffle(self.data_indexes[i]) 
               
            
    def get_cnts(self):
        list_cnt = []
        #list_min_class_cnt = []
        self.data_indexes = []
        for f in self.list_ann_files:
            temp_np = np.load(f)
            cnt_data = temp_np.shape[0] 
            
            if self.balanced_sampling:
                unique, counts = np.unique(temp_np, return_counts=True)                
                cnt_per_class = int(cnt_data * 0.2)

                selected_list = []
                for i in unique:
                    if i < 5:                        
                        selected = np.random.choice(np.where(temp_np == i)[0], cnt_per_class)                     
                    else:
                        selected = np.where(temp_np == 5)[0]
                    selected_list.append(selected)
                selected_list = np.concatenate(selected_list)
                
                self.data_indexes.append(selected_list) #The data used for training
                    
                list_cnt.append(len(selected_list))
                
            else:
                list_cnt.append(cnt_data)
                self.data_indexes.append(np.arange(cnt_data))
            
        self.list_cnt = list_cnt
        self.total_len = sum(list_cnt)  
        
        
        
class DataGenerator2(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_files, list_ann_files, 
                 batch_size=64, dim=(3000,1), n_classes=5, shuffle=True, prev_cnt = 10):
        # Constructor of the data generator.
        self.dim = dim
        self.batch_size = batch_size
        self.list_files = list_files
        self.list_ann_files = list_ann_files
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.prev_cnt = prev_cnt
        self.get_cnts() #Get the data count for each file        
        self.on_epoch_end() #Initialize file indexes        
        
    def __len__(self): 
        # Denotes the number of batches per epoch
        #return int((self.total_len+1) / self.batch_size)
        return sum(self.batch_per_file)
    
    def __getitem__(self, index):
        
        X = np.empty((self.batch_size,) + self.dim, dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.int32)

        curr_file_idx, accum_start, accum_end = self.get_accum_idx(index)
        
        curr_file = self.list_files[self.file_indexes[curr_file_idx]]
        curr_ann_file = self.list_ann_files[self.file_indexes[curr_file_idx]]        
        
        curr_np = np.load(curr_file)
        curr_ann = np.load(curr_ann_file)

        curr_batch = self.batch_indexes[self.file_indexes[curr_file_idx]]        

        end = min((self.batch_size - self.prev_cnt) * (curr_batch[index - accum_start]+1) + self.prev_cnt,
                    self.list_cnt[self.file_indexes[curr_file_idx]])
        start = end - self.batch_size        
   
        X = np.expand_dims(curr_np[start:end] , axis=-1)
        y = curr_ann[start:end]
        
        return X, y, curr_batch[index - accum_start]
    
    def get_accum_idx(self, index):
        curr_file_idx = 0
        accum_start = 0
        accum_end = self.batch_per_file[self.file_indexes[0]]
        for i in range(len(self.file_indexes)):
            if index < accum_end:
                curr_file_idx = i        
                break            
            accum_start += self.batch_per_file[self.file_indexes[i]]
            accum_end += self.batch_per_file[self.file_indexes[i+1]]
        
        return curr_file_idx, accum_start, accum_end
        
    def on_epoch_end(self):        
        self.curr_file_idx = 0
        # This function is called at the end of each epoch.
        self.file_indexes = np.arange(len(self.list_files)) #This is necessary to shuffle files
        self.batch_indexes = []
        self.batch_per_file = []
        for cnt in self.list_cnt:
            num_batches = 1 + (cnt - self.prev_cnt - 1) // (self.batch_size - self.prev_cnt)
            self.batch_per_file.append(num_batches)
            self.batch_indexes.append(np.arange(num_batches))
        
        #self.data_indexes = [np.arange(cnt) for cnt in self.list_cnt]
        if self.shuffle == True:
            np.random.shuffle(self.file_indexes)
            for i in range(len(self.batch_indexes)):
                np.random.shuffle(self.batch_indexes[i])
                #np.random.shuffle(self.data_indexes[i]) 
            
    def get_cnts(self):
        list_cnt = []
        for f in self.list_files:
            temp_np = np.load(f)
            cnt_data = temp_np.shape[0] 
            list_cnt.append(cnt_data)
            
        self.list_cnt = list_cnt
        self.total_len = sum(list_cnt)    