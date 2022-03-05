import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPool1D, ReLU, BatchNormalization, Lambda, Conv1D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l2


def bn_choice(bn=None):
    if bn is None:
        return Lambda(lambda x: x)
    elif bn == 'bn':
        return BatchNormalization(axis=-1)
    elif bn == 'sbn':
        return BatchNormalization(axis=-1, momentum=0)


def convK(in_planes: int, out_planes: int, kernel_size=16, strides: int = 1, groups: int = 1, l2_factor=1e-4):
    """Convolution 1D with kernel size > 1"""
    return Conv1D(
        out_planes,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        groups=groups,
        use_bias=True,
        kernel_initializer=VarianceScaling(),  
        kernel_regularizer=l2(l2_factor)
    )

def conv1(in_planes: int, out_planes: int, strides: int = 1, l2_factor=1e-4):
    """1x1 convolution"""
    return Conv1D(out_planes, kernel_size=1, strides=strides, use_bias=True, kernel_initializer=VarianceScaling(), kernel_regularizer=l2(l2_factor), padding='same')


class Bottleneck(tf.keras.layers.Layer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        strides: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        bn = None,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1(inplanes, width)
        self.bn1 = bn_choice(bn)
        self.conv2 = convK(width, width, strides, groups)
        self.bn2 = bn_choice(bn)
        self.conv3 = conv1(width, planes * self.expansion)
        self.bn3 = bn_choice(bn)
        if bn is not None:
            self.bn3.weight=0.0
        self.relu = ReLU()
        self.downsample = downsample
        self.strides = strides

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:            
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(tf.keras.layers.Layer):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        strides: int = 1,  
        downsample = None,      
        groups: int = 1,
        base_width: int = 64,      
        bn = None
    ) -> None:
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convK(inplanes, planes, strides=strides)
        self.bn1 = bn_choice(bn)
        self.relu = ReLU()
        self.conv2 = convK(planes, planes)
        self.bn2 = bn_choice(bn)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if bn is not None:            
            self.bn2.weight=0.0

        self.downsample = downsample
        self.strides = strides

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(tf.keras.Model):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 6,        
        groups: int = 1,
        width_per_group: int = 64,
        bn = None,
        l2_factor = 1e-4,
        input_shape=(3000,1),
        pool_list=[2,2,2,4]
    ) -> None:
        super().__init__()        
        norm_layer = bn_choice(bn)
        self._bn = bn

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv1D(self.inplanes, kernel_size=3, strides=1, padding='same', use_bias=True,
                kernel_initializer=VarianceScaling(),  kernel_regularizer=l2(l2_factor), input_shape=input_shape)
        self.bn1 = norm_layer
        self.relu = ReLU()        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], strides=pool_list[0])
        self.layer3 = self._make_layer(block, 256, layers[2], strides=pool_list[1])
        self.layer4 = self._make_layer(block, 512, layers[3], strides=pool_list[2])
        self.maxpool = Sequential([MaxPool1D(pool_size=pool_list[3]), Flatten()])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Dense(num_classes, use_bias=True, activation='softmax')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, Bottleneck):
        #            nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #        elif isinstance(m, BasicBlock):
        #            nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        strides: int = 1
    ):
        norm_layer = bn_choice(self._bn)
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential([
                conv1(self.inplanes, planes * block.expansion, strides),                
                norm_layer
                #MaxPool2D(pool_size=strides)
                #norm_layer(planes * block.expansion),
            ])

        layers = []
        layers.append(
            block(
                self.inplanes, planes, strides, downsample, self.groups, self.base_width, self._bn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    bn=self._bn,
                )
            )

        return Sequential(layers)

    def call(self, x):                
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)        
        
        #x = self.maxpool(x)

        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)     
        x = self.maxpool(x)  
        x = self.fc(x)

        return x


def ResNet34(input_shape, bn='bn', pool_list=[2,2,2,2], num_classes=6):
    return ResNet(BasicBlock, [4,4,4,4], input_shape=input_shape, bn=bn, pool_list=pool_list, num_classes=num_classes)
