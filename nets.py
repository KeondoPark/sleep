import tensorflow as tf
from tensorflow.keras.layers \
    import BatchNormalization, Conv1D, Conv2D, ReLU, Input, Dense, Flatten, RepeatVector, Reshape, Dropout, add,\
        MaxPool1D, MaxPool2D, GlobalAveragePooling1D, GlobalMaxPooling1D

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
        qk = tf.keras.backend.softmax(qk) * self.scaling # (B, h, n, n)            
        attn = qk / self.scaling
        
        output = tf.matmul(attn, v) # (B, h, n, d)
        output = tf.transpose(output, perm=[0,2,1,3]) #(B, n, h, d)
        output = Reshape((-1, self.embed_dim * self.n_heads))(output)
        return  output
        
class conv1d_block(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=100, strides=1, padding='valid'):
        super().__init__()
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)
        self.bn = BatchNormalization(axis=-1)
        self.relu = ReLU()
    def call(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Conv1DAttention(tf.keras.Model):
    def __init__(self, input_shape=(3000,1), n_classes=6):
        super().__init__()
        #self.input = Input(shape=input_shape)
        self.conv0_1 = conv1d_block(filters=32, kernel_size=300, strides=5)
        self.conv0_2 = conv1d_block(filters=64, kernel_size=5, strides=3)
        self.squeeze_conv1 = conv1d_block(filters=64, kernel_size=179, strides=1)
        self.squeeze_conv2 = conv1d_block(filters=256, kernel_size=1, strides=1)
        self.mha1 = MultiheadAttention(n_heads=8, embed_dim=32)
        self.conv2_1 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv2_2 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv2_3 = conv1d_block(filters=256, kernel_size=1, strides=1, padding='same')
        self.dropout2_1 = Dropout(0.2)
        
        #self.mha2 = MultiheadAttention(n_heads=8, embed_dim=32)
        #self.conv2_4 = conv1d_block(filters=512, kernel_size=1, strides=1, padding='same')
        #self.conv2_5 = conv1d_block(filters=512, kernel_size=1, strides=1, padding='same')
        #self.conv2_6 = conv1d_block(filters=512, kernel_size=1, strides=1, padding='same')        
        
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
        #y = self.squeeze_conv1(x)
        #y = self.squeeze_conv2(y)
        #identity = y
        #y = self.mha1(y)
        #y = self.conv2_1(y)
        #y = self.conv2_2(y)
        #y = self.conv2_3(y)
        #y = self.dropout2_1(y)
        #y = add([identity, y])
        
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
        
        
        #y = Reshape((256,))(y)
        #y = RepeatVector(42)(y)
        
        #x = add([x, y])
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
        self.conv0_1 = conv1d_block(filters=32, kernel_size=5, strides=3)
        self.conv0_2 = conv1d_block(filters=32, kernel_size=3, strides=1)
        self.conv0_3 = conv1d_block(filters=32, kernel_size=10, strides=5)

        self.maxpool0_1 = MaxPool1D(pool_size=3)
        self.maxpool0_2 = MaxPool1D(pool_size=2)
        self.maxpool0_3 = MaxPool1D(pool_size=3)

        self.conv1_1 = conv1d_block(filters=64, kernel_size=3, strides=3)
        self.conv1_2 = conv1d_block(filters=64, kernel_size=15, strides=15)
        self.conv1_3 = conv1d_block(filters=64, kernel_size=3, strides=2)

        self.conv2_1 = conv1d_block(filters=128, kernel_size=3, strides=2)
        self.conv2_2 = conv1d_block(filters=128, kernel_size=3, strides=2)
        self.conv2_3 = conv1d_block(filters=128, kernel_size=3, strides=2)

        self.conv3_1 = conv1d_block(filters=256, kernel_size=3, strides=2)
        self.conv3_2 = conv1d_block(filters=256, kernel_size=3, strides=2)
        self.conv3_3 = conv1d_block(filters=256, kernel_size=3, strides=2)

        self.gpool1_1 = GlobalAveragePooling1D()
        self.gpool1_2 = GlobalAveragePooling1D()
        self.gpool1_3 = GlobalAveragePooling1D()

        self.mha1 = MultiheadAttention(n_heads=8, embed_dim=32)
        self.mha2 = MultiheadAttention(n_heads=8, embed_dim=32)
        self.mha3 = MultiheadAttention(n_heads=8, embed_dim=32)

        self.conv4_1 = conv1d_block(filters=256, kernel_size=1, strides=1)
        self.conv4_2 = conv1d_block(filters=256, kernel_size=1, strides=1)
        self.conv4_3 = conv1d_block(filters=256, kernel_size=1, strides=1)

        self.conv5_1 = conv1d_block(filters=256, kernel_size=1, strides=1)
        
        
        self.fc = Dense(n_classes, activation='softmax')
        
    def call(self, x):

        x1 = self.conv0_1(x)
        x2 = self.conv0_2(x)
        x3 = self.conv0_3(x)

        x1 = self.maxpool0_1(x1)
        x2 = self.maxpool0_2(x2)
        x3 = self.maxpool0_3(x3)

        x1 = self.conv1_1(x1)
        x2 = self.conv1_2(x2)
        x3 = self.conv1_3(x3)

        x1 = self.conv2_1(x1)
        x2 = self.conv2_2(x2)
        x3 = self.conv2_3(x3)

        x1 = self.conv3_1(x1)    
        x2 = self.conv3_2(x2)    
        x3 = self.conv3_3(x3)

        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)

        y1 = self.gpool1_1(x1)
        y2 = self.gpool1_2(x2)
        y3 = self.gpool1_3(x3)            

        y1 = Reshape((-1,256))(y1)
        y2 = Reshape((-1,256))(y2)
        y3 = Reshape((-1,256))(y3)

        identity1 = y1
        identity2 = y2
        identity3 = y3

        y1 = self.mha1(y1)
        y2 = self.mha2(y2)
        y3 = self.mha3(y3)

        y1 = add([identity1, y1])
        y2 = add([identity2, y2])
        y3 = add([identity3, y3])

        identity1 = y1
        identity2 = y2
        identity3 = y3

        y1 = self.conv4_1(y1)
        y2 = self.conv4_2(y2)
        y3 = self.conv4_3(y3)

        y1 = add([identity1, y1])
        y2 = add([identity2, y2])
        y3 = add([identity3, y3])

        y1 = tf.tile(y1, [1, x1.shape[1], 1])
        y2 = tf.tile(y2, [1, x2.shape[1], 1])
        y3 = tf.tile(y3, [1, x3.shape[1], 1])

        x1 = x1 + y1
        x2 = x2 + y2
        x3 = x3 + y3

        x_concat = tf.concat([x1,x2,x3], axis=1)

        x = self.conv5_1(x_concat)    
        
        x = Flatten()(x)
        x = self.fc(x)
        
        return x


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
        


