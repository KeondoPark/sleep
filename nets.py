#from tkinter import Y
import tensorflow as tf
from tensorflow.keras.layers \
    import BatchNormalization, Conv1D, Conv2D, ReLU, Input, Dense, Flatten, RepeatVector, Reshape, Dropout, add,\
        MaxPool1D, MaxPool2D, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.emb = Dense(embed_dim * n_heads * 3, use_bias=True)         
        d = tf.cast(embed_dim, dtype=tf.float32)    
        self.scaling = 1/tf.math.sqrt(d)

    def call(self, inputs):    
        """
        inputs: (B, num_seed, features)
        """
        #num_seed = tf.shape(inputs)[1]
        embedding = self.emb(inputs) # (B, n, d * h * 3)            
        heads = Reshape((-1, self.embed_dim, self.n_heads, 3))(embedding) #(B, n, d, h, 3)
        

        heads = tf.transpose(heads, perm=[0,4,3,1,2]) # (B, 3, h, n, d)
        q = heads[:,0,:,:,:] #(B, h, n, d)
        k = heads[:,1,:,:,:] #(B, h, n, d)
        v = heads[:,2,:,:,:] #(B, h, n, d)
        
        qk = tf.matmul(q, k, transpose_b=True) # (B, h, n, n)    
        attn = tf.keras.backend.softmax(qk) * self.scaling # (B, h, n, n)            
        #attn = qk / self.scaling
        
        output = tf.matmul(attn, v) # (B, h, n, d)
        output = tf.transpose(output, perm=[0,2,1,3]) #(B, n, h, d)
        output = Reshape((-1, self.embed_dim * self.n_heads))(output)
        return  output

class MultiheadAttention_Feat(tf.keras.layers.Layer):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.emb = Dense(embed_dim * n_heads * 3, use_bias=True)         
        d = tf.cast(embed_dim, dtype=tf.float32)    
        self.scaling = 1.0 # We will use n=1 in this attention module

    def call(self, inputs):    
        """
        inputs: (B, num_seed, features)
        """
        #num_seed = tf.shape(inputs)[1]
        embedding = self.emb(inputs) # (B, n, d * h * 3)            
        heads = Reshape((-1, self.embed_dim, self.n_heads, 3))(embedding) #(B, n, d, h, 3)
        

        heads = tf.transpose(heads, perm=[0,4,3,2,1]) # (B, 3, h, n, d)
        q = heads[:,0,:,:,:] #(B, h, d, n)
        k = heads[:,1,:,:,:] #(B, h, d, n)
        v = heads[:,2,:,:,:] #(B, h, d, n)
        
        qk = tf.matmul(q, k, transpose_b=True) # (B, h, d, d)    
        attn = tf.keras.backend.softmax(qk) * self.scaling # (B, h, d, d)            
        #attn = qk / self.scaling
        
        output = tf.matmul(attn, v) # (B, h, d, n)
        output = tf.transpose(output, perm=[0,3,1,2]) #(B, n, h, d)
        output = Reshape((-1, self.embed_dim * self.n_heads))(output)
        return  output
        
class conv1d_block(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=100, strides=1, padding='valid', dilation_rate=1):
        super().__init__()
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate)
        self.bn = BatchNormalization(axis=-1)
        self.relu = ReLU()
    def call(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Conv1DAttention(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()
        #self.input = Input(shape=input_shape)
        self.conv0_1 = conv1d_block(filters=32, kernel_size=300, strides=5, padding='same')
        self.conv0_2 = conv1d_block(filters=64, kernel_size=5, strides=3, padding='same')
        self.squeeze_conv1 = conv1d_block(filters=64, kernel_size=200, strides=1)
        self.squeeze_conv2 = conv1d_block(filters=256, kernel_size=1, strides=1)
        self.mha1 = MultiheadAttention(n_heads=8, embed_dim=32)
        self.conv2_1 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv2_2 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv2_3 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.dropout2_1 = Dropout(0.2)
        
        self.conv1_1 = conv1d_block(filters=128, kernel_size=5, strides=2)
        self.conv1_2 = conv1d_block(filters=128, kernel_size=5, strides=1, padding='same')
        self.conv1_3 = conv1d_block(filters=128, kernel_size=5, strides=1, padding='same')
        self.dropout1_1 = Dropout(0.2)
        
        self.conv1_4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv1_5 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.conv1_6 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.dropout1_2 = Dropout(0.2)
                
        self.dropout3 = Dropout(0.2)
        self.final_conv1 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.final_conv2 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.final_conv3 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        
        self.fc = Dense(n_classes, activation='softmax')
        
    def call(self, x):
        #x = self.input(x)
        x = self.conv0_1(x)
        x = self.conv0_2(x)
        y = self.squeeze_conv1(x)
        y = self.squeeze_conv2(y)
        identity = y
        y = self.mha1(y)
        y = self.conv2_1(y)
        y = self.conv2_2(y)
        y = self.conv2_3(y)
        y = self.dropout2_1(y)
        y = add([identity, y])
        
        x = self.conv1_1(x)
        identity = x
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.dropout1_1(x)
        x = add([identity, x])
        
        x = self.conv1_4(x)
        identity = x
        x = self.conv1_5(x)
        x = self.conv1_6(x)
        x = self.dropout1_2(x)
        x = add([identity, x])
        
        
        y = Reshape((256,))(y)
        y = RepeatVector(x.shape[1])(y)
        
        x = add([x, y])
        #x = y
        #x = self.dropout3(x)
               
        identity = x
        x = self.final_conv1(x)        
        x = self.final_conv2(x)        
        x = self.final_conv3(x) 
        x = add([identity, x])
        x = Flatten()(x)
        x = self.fc(x)
        
        return x


class Conv1DAttention2(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()
        #self.input = Input(shape=input_shape)
        self.conv0_1 = conv1d_block(filters=32, kernel_size=300, strides=5, padding='same')
        self.conv0_2 = conv1d_block(filters=64, kernel_size=5, strides=3, padding='same')
        self.squeeze_conv1 = conv1d_block(filters=64, kernel_size=200, strides=1)
        self.squeeze_conv2 = conv1d_block(filters=256, kernel_size=1, strides=1)
        self.mha1 = MultiheadAttention_Feat(n_heads=8, embed_dim=32)
        self.conv2_1 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv2_2 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv2_3 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.dropout2_1 = Dropout(0.2)
        
        self.conv1_1 = conv1d_block(filters=128, kernel_size=5, strides=2)
        self.conv1_2 = conv1d_block(filters=128, kernel_size=5, strides=1, padding='same')
        self.conv1_3 = conv1d_block(filters=128, kernel_size=5, strides=1, padding='same')
        self.dropout1_1 = Dropout(0.2)
        
        self.conv1_4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv1_5 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.conv1_6 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.dropout1_2 = Dropout(0.2)
                
        self.dropout3 = Dropout(0.2)
        self.final_conv1 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.final_conv2 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        self.final_conv3 = conv1d_block(filters=256, kernel_size=5, strides=1, padding='same')
        
        self.fc = Dense(n_classes, activation='softmax')
        
    def call(self, x):
        #x = self.input(x)
        x = self.conv0_1(x)
        x = self.conv0_2(x)
        y = self.squeeze_conv1(x)
        y = self.squeeze_conv2(y)
        identity = y
        y = self.mha1(y)
        y = self.conv2_1(y)
        y = self.conv2_2(y)
        y = self.conv2_3(y)
        y = self.dropout2_1(y)
        y = add([identity, y])
        
        x = self.conv1_1(x)
        identity = x
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.dropout1_1(x)
        x = add([identity, x])
        
        x = self.conv1_4(x)
        identity = x
        x = self.conv1_5(x)
        x = self.conv1_6(x)
        x = self.dropout1_2(x)
        x = add([identity, x])
        
        
        y = Reshape((256,))(y)
        y = RepeatVector(x.shape[1])(y)
        
        x = add([x, y])
        #x = y
        #x = self.dropout3(x)
               
        identity = x
        x = self.final_conv1(x)        
        x = self.final_conv2(x)        
        x = self.final_conv3(x) 
        x = add([identity, x])
        x = Flatten()(x)
        x = self.fc(x)
        
        return x
#Previously named as Conv1dAttention3
class Conv1DASPP(tf.keras.Model):
    
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()

        self.conv0 = conv1d_block(filters=32, kernel_size=5, padding='same')
        self.conv1 = conv1d_block(filters=64, kernel_size=5, padding='same')
        self.conv2 = conv1d_block(filters=64, kernel_size=5, padding='same')

        #ASPP
        self.conv3_1 = conv1d_block(filters=64, kernel_size=10, dilation_rate=1, padding='same')
        self.conv3_2 = conv1d_block(filters=64, kernel_size=10, dilation_rate=2, padding='same')
        self.conv3_3 = conv1d_block(filters=64, kernel_size=10, dilation_rate=4, padding='same')
        self.conv3_4 = conv1d_block(filters=64, kernel_size=10, dilation_rate=8, padding='same')
        self.gpool = GlobalAveragePooling1D()
        self.conv3_5_1 = Conv1D(filters=64, kernel_size=1)
        self.conv3_5_bn = BatchNormalization(axis=-1)

        self.maxpool3 = MaxPool1D(pool_size=10)
        #self.dropout3 = Dropout(0.2)

        self.conv4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv5 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv6 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout6 = Dropout(0.2)

        self.conv7 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv8 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv9 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout9 = Dropout(0.2)

        self.fc = Dense(n_classes, activation='softmax')

        
    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)
        x3 = self.conv3_3(x)
        x4 = self.conv3_4(x)
        x5 = self.gpool(x)
        x5 = Reshape((1,64))(x5)
        x5 = self.conv3_5_1(x5)
        x5 = self.conv3_5_bn(x5)
        x5 = tf.tile(x5, [1,x.shape[1],1])        

        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)

        x = self.maxpool3(x)
        
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = add([identity, x])
        x = self.dropout6(x)

        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = add([identity, x])
        x = self.dropout9(x)

        x = Flatten()(x)
        x = self.fc(x)        

        return x


class Conv1DASPP_1(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()

        self.conv0 = conv1d_block(filters=32, kernel_size=5, padding='same')
        self.conv1 = conv1d_block(filters=64, kernel_size=5, padding='same')
        self.conv2 = conv1d_block(filters=64, kernel_size=5, padding='same')

        #ASPP
        self.conv3_1 = conv1d_block(filters=64, kernel_size=10, dilation_rate=1, padding='same')
        self.conv3_2 = conv1d_block(filters=64, kernel_size=10, dilation_rate=2, padding='same')        
        self.gpool = GlobalAveragePooling1D()
        self.conv3_5_1 = Conv1D(filters=64, kernel_size=1)
        self.conv3_5_bn = BatchNormalization(axis=-1)

        self.maxpool3 = MaxPool1D(pool_size=10)
        #self.dropout3 = Dropout(0.2)

        self.conv4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv5 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv6 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout6 = Dropout(0.2)

        self.conv7 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv8 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv9 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout9 = Dropout(0.2)

        self.fc = Dense(n_classes, activation='softmax')

        
    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)        
        x5 = self.gpool(x)
        x5 = Reshape((1,64))(x5)
        x5 = self.conv3_5_1(x5)
        x5 = self.conv3_5_bn(x5)
        x5 = tf.tile(x5, [1,x.shape[1],1])        

        x = tf.concat([x1, x2, x5], axis=-1)        

        x = self.maxpool3(x)
        
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = add([identity, x])
        x = self.dropout6(x)

        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = add([identity, x])
        x = self.dropout9(x)

        x = Flatten()(x)
        x = self.fc(x)        

        return x


class Conv1DASPP_2(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()

        self.conv0 = conv1d_block(filters=32, kernel_size=5, padding='same')
        self.conv1 = conv1d_block(filters=64, kernel_size=5, padding='same')
        self.conv2 = conv1d_block(filters=64, kernel_size=5, padding='same')

        #ASPP
        self.conv3_1 = conv1d_block(filters=64, kernel_size=10, dilation_rate=1, padding='same')
        self.conv3_2 = conv1d_block(filters=64, kernel_size=10, dilation_rate=2, padding='same')
        self.gpool = GlobalAveragePooling1D()
        self.mha1 = MultiheadAttention_Feat(n_heads=2, embed_dim=32)
        self.conv3_5_1 = Conv1D(filters=64, kernel_size=1)
        self.conv3_5_bn = BatchNormalization(axis=-1)

        self.maxpool3 = MaxPool1D(pool_size=10)
        #self.dropout3 = Dropout(0.2)

        self.conv4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv5 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv6 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout6 = Dropout(0.2)

        self.conv7 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv8 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv9 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout9 = Dropout(0.2)

        self.fc = Dense(n_classes, activation='softmax')

        
    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)        
        x5 = self.gpool(x)
        x5 = Reshape((1,64))(x5)
        x5 = self.mha1(x5)
        x5 = self.conv3_5_1(x5)
        x5 = self.conv3_5_bn(x5)
        x5 = tf.tile(x5, [1,x.shape[1],1])        

        x = tf.concat([x1, x2, x5], axis=-1)        

        x = self.maxpool3(x)
        
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = add([identity, x])
        x = self.dropout6(x)

        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = add([identity, x])
        x = self.dropout9(x)

        x = Flatten()(x)
        x = self.fc(x)        

        return x    

class Conv1DASPP_3(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()

        self.conv0 = conv1d_block(filters=32, kernel_size=5, padding='same')
        self.conv1 = conv1d_block(filters=64, kernel_size=5, padding='same')
        self.conv2 = conv1d_block(filters=64, kernel_size=5, padding='same')

        #ASPP
        self.conv3_1 = conv1d_block(filters=64, kernel_size=10, dilation_rate=1, padding='same')
        self.conv3_2 = conv1d_block(filters=64, kernel_size=10, dilation_rate=2, padding='same')
        self.conv3_3 = conv1d_block(filters=64, kernel_size=10, dilation_rate=4, padding='same')
        self.conv3_4 = conv1d_block(filters=64, kernel_size=10, dilation_rate=16, padding='same')
        self.conv3_5 = conv1d_block(filters=64, kernel_size=10, dilation_rate=32, padding='same')
        self.gpool = GlobalAveragePooling1D()
        self.mha1 = MultiheadAttention_Feat(n_heads=2, embed_dim=32)
        self.conv3_5_1 = Conv1D(filters=64, kernel_size=1)
        self.conv3_5_bn = BatchNormalization(axis=-1)

        self.maxpool3 = MaxPool1D(pool_size=10)
        #self.dropout3 = Dropout(0.2)

        self.conv4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv5 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv6 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout6 = Dropout(0.2)

        self.conv7 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv8 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv9 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout9 = Dropout(0.2)

        self.fc = Dense(n_classes, activation='softmax')

        
    def call(self, x, out_mode='prob'):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)        
        x3 = self.conv3_3(x)        
        x4 = self.conv3_4(x)        
        x5 = self.conv3_5(x)
        x6 = self.gpool(x)
        x6 = Reshape((1,64))(x6)
        x6 = self.mha1(x6)
        x6 = self.conv3_5_1(x6)
        x6 = self.conv3_5_bn(x6)
        x6 = tf.tile(x6, [1,x.shape[1],1])        

        x = tf.concat([x1, x2, x3, x4, x5, x6], axis=-1)        

        x = self.maxpool3(x)
        
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = add([identity, x])
        x = self.dropout6(x)

        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = add([identity, x])
        x = self.dropout9(x)

        if out_mode == 'feature':
            return  x
        else:
            x = Flatten()(x)
            x = self.fc(x)                
            return x

class Conv1D_SPP(tf.keras.Model):
    
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()

        self.conv0 = conv1d_block(filters=32, kernel_size=5, padding='same')
        self.conv1 = conv1d_block(filters=64, kernel_size=5, padding='same')
        self.conv2 = conv1d_block(filters=64, kernel_size=5, padding='same')

        #ASPP
        self.maxpool3_1 = MaxPool1D(pool_size=10, strides=2, padding='same')
        self.maxpool3_2 = MaxPool1D(pool_size=20, strides=2, padding='same')
        #self.maxpool3_3 = MaxPool1D(pool_size=40, stride=1, padding='same')
        #self.maxpool3_4 = MaxPool1D(pool_size=80, stride=1, padding='same')
        self.gpool = GlobalAveragePooling1D()
        self.conv3_5_1 = Conv1D(filters=64, kernel_size=1)
        self.conv3_5_bn = BatchNormalization(axis=-1)

        #self.maxpool3 = MaxPool1D(pool_size=10)
        #self.dropout3 = Dropout(0.2)

        self.conv4 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv5 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv6 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout6 = Dropout(0.2)

        self.conv7 = conv1d_block(filters=256, kernel_size=5, strides=2)
        self.conv8 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv9 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout9 = Dropout(0.2)

        self.fc = Dense(n_classes, activation='softmax')

        
    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.maxpool3_1(x)
        x2 = self.maxpool3_2(x)
        #x3 = self.maxpool3_3(x)
        #x4 = self.maxpool3_4(x)
        x5 = self.gpool(x)
        x5 = Reshape((1,64))(x5)
        x5 = self.conv3_5_1(x5)
        x5 = self.conv3_5_bn(x5)
        x5 = tf.tile(x5, [1,x1.shape[1],1])        

        x = tf.concat([x1, x2, x5], axis=-1)
        
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = add([identity, x])
        x = self.dropout6(x)

        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = add([identity, x])
        x = self.dropout9(x)

        x = Flatten()(x)
        x = self.fc(x)                
        return x

class Conv1DASPPLayer(tf.keras.layers.Layer):
    def __init__(self, dil_fac=1):
        super().__init__()

        self.conv0 = conv1d_block(filters=32, kernel_size=5, padding='same')
        self.conv1 = conv1d_block(filters=64, kernel_size=5, padding='same')
        self.conv2 = conv1d_block(filters=64, kernel_size=5, padding='same')

        #ASPP
        self.conv3_1 = conv1d_block(filters=64, kernel_size=10, dilation_rate=1 * dil_fac, padding='same')
        self.conv3_2 = conv1d_block(filters=64, kernel_size=10, dilation_rate=2 * dil_fac, padding='same')
        self.conv3_3 = conv1d_block(filters=64, kernel_size=10, dilation_rate=4 * dil_fac, padding='same')
        self.conv3_4 = conv1d_block(filters=64, kernel_size=10, dilation_rate=16 * dil_fac, padding='same')
        self.conv3_5 = conv1d_block(filters=64, kernel_size=10, dilation_rate=32 * dil_fac, padding='same')
        self.gpool = GlobalAveragePooling1D()
        self.mha1 = MultiheadAttention_Feat(n_heads=2, embed_dim=32)
        self.conv3_5_1 = Conv1D(filters=64, kernel_size=1)
        self.conv3_5_bn = BatchNormalization(axis=-1)

        self.maxpool3 = MaxPool1D(pool_size=10)
        #self.dropout3 = Dropout(0.2)

        self.conv4 = conv1d_block(filters=256, kernel_size=5, strides=3)
        self.conv5 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv6 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout6 = Dropout(0.2)

        self.conv7 = conv1d_block(filters=256, kernel_size=5, strides=3)
        self.conv8 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv9 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout9 = Dropout(0.2)

        self.conv10 = conv1d_block(filters=256, kernel_size=5, strides=3)
        self.conv11 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.conv12 = conv1d_block(filters=256, kernel_size=5, padding='same')
        self.dropout12 = Dropout(0.2)
    
    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)        
        x3 = self.conv3_3(x)        
        x4 = self.conv3_4(x)        
        x5 = self.conv3_5(x)
        x6 = self.gpool(x)
        x6 = Reshape((1,64))(x6)
        x6 = self.mha1(x6)
        x6 = self.conv3_5_1(x6)
        x6 = self.conv3_5_bn(x6)
        x6 = tf.tile(x6, [1,x.shape[1],1])        

        x = tf.concat([x1, x2, x3, x4, x5, x6], axis=-1)        

        x = self.maxpool3(x)
        
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = add([identity, x])
        x = self.dropout6(x)

        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = self.conv9(x)
        x = add([identity, x])
        x = self.dropout9(x)

        x = self.conv10(x)
        identity = x
        x = self.conv11(x)
        x = self.conv12(x)
        x = add([identity, x])
        x = self.dropout12(x)

        return x


class Conv1DASPP_single(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6, dil_fac=1):
        super().__init__()
        self.aspp = Conv1DASPPLayer(dil_fac=dil_fac)        

        self.fc = Dense(n_classes, activation='softmax')
        
    def call(self, x):        

        x = self.aspp(x)
        x = Flatten()(x)
        out = self.fc(x)

        return out

class Conv1DASPP_multi(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6, batch_size=64, prev_cnt=10):
        super().__init__()
        self.aspp = Conv1DASPPLayer()
        self.batch_size = batch_size
        self.prev_cnt = prev_cnt
        #self.conv = conv1d_block(filters=256, kernel_size=10, padding='valid')
        
        #self.lstm = LSTM(64)
        self.fc = Dense(n_classes, activation='softmax')
        
    def call(self, x):
        embeddings = []
        for i in range(self.prev_cnt + 1):
            single_epoch = self.aspp(x[:,i])
            embeddings.append(single_epoch)

        embeddings = tf.concat(embeddings, axis=1)
        out = self.fc(Flatten()(embeddings))
        '''
        x = self.conv(x)
        x = Reshape((256,))(x)
        x_stack = []
        for i in range(self.prev_cnt):
            x_bfi = x[:i]
            #x_i = tf.tile(x[i,None], [self.prev_cnt + 1 -i,1,1])
            x_i = tf.tile(x[i,None], [self.prev_cnt + 1 -i,1])
            x_new = tf.concat([x_bfi, x_i], axis=0)
            x_stack.append(x_new)

        for i in range(self.batch_size - self.prev_cnt):
            x_stack.append(x[i:i+self.prev_cnt + 1])

        x_stack = tf.stack(x_stack)
        
        x = self.lstm(x_stack)
        x = Flatten()(x)
        out = self.fc(x)
        '''
        return out

class conv2d_block(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=(3,3), strides=(1,1), padding='valid'):
        super().__init__()
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)
        self.bn = BatchNormalization(axis=-1)
        self.relu = ReLU()
    def call(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Conv2DSimple(tf.keras.Model):
    def __init__(self, input_shape=(3000,35,1), n_classes=6):
        super().__init__()
        #self.input = Input(shape=input_shape)
        self.conv0_1 = conv2d_block(filters=32, kernel_size=(10,5), strides=(3,1))        
        self.maxpool0_1 = MaxPool2D(pool_size=(1,3))        

        self.conv1_1 = conv2d_block(filters=64, kernel_size=(10,5), strides=(3,1))
        self.maxpool1_1 = MaxPool2D(pool_size=(1,3))        
        

        self.conv2_1 = conv2d_block(filters=128, kernel_size=(10,1), strides=(3,1))
        self.conv3_1 = conv2d_block(filters=256, kernel_size=(10,1), strides=(3,1))
        self.conv4_1 = conv2d_block(filters=256, kernel_size=(1,1), strides=(1,1))
        self.conv5_1 = conv2d_block(filters=256, kernel_size=(1,1), strides=(1,1))
        
        self.fc = Dense(n_classes, activation='softmax')
        
    def call(self, x):

        x = self.conv0_1(x)
        x = self.maxpool0_1(x)        

        x = self.conv1_1(x)
        x = self.maxpool1_1(x)        

        x = self.conv2_1(x)
        x = self.conv3_1(x)
        x = self.conv4_1(x)
        x = self.conv5_1(x)
        
        x = Flatten()(x)
        x = self.fc(x)
        
        return x
        


