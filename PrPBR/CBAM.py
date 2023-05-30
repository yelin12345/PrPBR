import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Reshape, Dropout, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPool2D, Concatenate


# 继承Layer,建立resnet50 101 152卷积层模块
def conv_block(inputs, filter_num, reduction_ratio, stride=1, name=None):
    x = inputs
    x = Conv2D(filter_num[0], (1, 1), strides=stride, padding='same', name=name + '_conv1')(x)
    x = BatchNormalization(axis=3, name=name + '_bn1')(x)
    x = Activation('relu', name=name + '_relu1')(x)

    x = Conv2D(filter_num[1], (3, 3), strides=1, padding='same', name=name + '_conv2')(x)
    x = BatchNormalization(axis=3, name=name + '_bn2')(x)
    x = Activation('relu', name=name + '_relu2')(x)

    x = Conv2D(filter_num[2], (1, 1), strides=1, padding='same', name=name + '_conv3')(x)
    x = BatchNormalization(axis=3, name=name + '_bn3')(x)

    # Channel Attention
    avgpool = GlobalAveragePooling2D(name=name + '_channel_avgpool')(x)
    maxpool = GlobalMaxPool2D(name=name + '_channel_maxpool')(x)
    # Shared MLP
    Dense_layer1 = Dense(filter_num[2] // reduction_ratio, activation='relu', name=name + '_channel_fc1')
    Dense_layer2 = Dense(filter_num[2], activation='relu', name=name + '_channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = layers.add([avg_out, max_out])
    channel = Activation('sigmoid', name=name + '_channel_sigmoid')(channel)
    channel = Reshape((1, 1, filter_num[2]), name=name + '_channel_reshape')(channel)
    channel_out = tf.multiply(x, channel)

    # Spatial Attention
    avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, name=name + '_spatial_avgpool')
    maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, name=name + '_spatial_maxpool')
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (7, 7), strides=1, padding='same', name=name + '_spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name=name + '_spatial_sigmoid')(spatial)

    CBAM_out = tf.multiply(channel_out, spatial_out)

    # residual connection
    r = Conv2D(filter_num[2], (1, 1), strides=stride, padding='same', name=name + '_residual')(inputs)
    x = layers.add([CBAM_out, r])
    x = Activation('relu', name=name + '_relu3')(x)

    return x


def build_block(x, filter_num, blocks, reduction_ratio=16, stride=1, name=None):
    x = conv_block(x, filter_num, reduction_ratio, stride, name=name)

    for i in range(1, blocks):
        x = conv_block(x, filter_num, reduction_ratio, stride=1, name=name + '_block' + str(i))

    return x
