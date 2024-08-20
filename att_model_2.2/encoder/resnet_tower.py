import tensorflow as tf


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [0, 0],
                                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end],
                                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d(inputs, filters, kernel_size, strides, data_format):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.keras.layers.Conv2D(filters, kernel_size,
                                  strides=strides,
                                  padding='SAME',
                                  use_bias=False,
                                  kernel_initializer=tf.initializers.VarianceScaling(scale=2.0),
                                  kernel_regularizer=None,
                                  data_format=data_format)(inputs)


def bn(inputs, data_format):
    bn_axis = 1 if data_format == 'channels_first' else 3
    return tf.keras.layers.BatchNormalization(axis=bn_axis,
                                              momentum=0.9997,
                                              scale=True,
                                              center=True,
                                              fused=True)(inputs)


def relu(inputs):
    return tf.keras.layers.Activation('relu')(inputs)


def conv2d_bn_relu(inputs, filters, kernel_size, strides, data_format):
    inputs = conv2d(inputs, filters, kernel_size, strides, data_format)
    inputs = bn(inputs, data_format)
    return relu(inputs)


def conv2d_bn(inputs, filters, kernel_size, strides, data_format):
    inputs = conv2d(inputs, filters, kernel_size, strides, data_format)
    return bn(inputs, data_format)


def _resnet_block(inputs, depth, block_num, data_format):
    net = inputs
    for _ in range(block_num):
        axis = 1 if data_format == "channels_first" else 3
        if net.get_shape().dims[axis] != depth:
            shortcut = conv2d_bn(net, depth, 1, 1, data_format)
        else:
            shortcut = net
        net = conv2d_bn_relu(net, depth, 3, 1, data_format)
        net = conv2d_bn(net, depth, 3, 1, data_format)
        net = relu(net + shortcut)
    return net


def resnet_tower(inputs, depth_rate=1.0, min_depth=16, blocks=[1, 2, 5, 3], for_att=False, data_format="channels_last"):
    def depth(dp):
        dp = int(dp * depth_rate)
        return dp if dp >= min_depth else min_depth

    net = inputs
    # block 1
    with tf.name_scope("block_0"):
        net = conv2d_bn_relu(net, depth(32), 3, 1, data_format)
        net = conv2d_bn_relu(net, depth(64), 3, 1, data_format)

    if len(blocks) <= 0:
        return net
    # block 2
    with tf.name_scope("block_1"):
        net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, data_format=data_format)(net)
        net = _resnet_block(net, depth(128), blocks[0], data_format)
        net = conv2d_bn_relu(net, depth(128), 3, 1, data_format)

    if len(blocks) <= 1:
        return net
    # block 3
    with tf.name_scope("block_2"):
        net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, data_format=data_format)(net)
        net = _resnet_block(net, depth(256), blocks[1], data_format)
        net = conv2d_bn_relu(net, depth(256), 3, 1, data_format)

    if len(blocks) <= 2:
        return net
    # block 4
    with tf.name_scope("block_3"):
        strides = [2, 2] if for_att else [2, 1]
        net = tf.keras.layers.MaxPool2D(pool_size=2, strides=strides, data_format=data_format)(net)
        net = _resnet_block(net, depth(512), blocks[2], data_format)
        net = conv2d_bn_relu(net, depth(512), 3, 1, data_format)

    if len(blocks) <= 3:
        return net
    # block 5
    with tf.name_scope("block_4"):
        net = _resnet_block(net, depth(512), blocks[3], data_format)
        net = conv2d_bn_relu(net, depth(512), [3, 3], 1, data_format)
        net = conv2d_bn_relu(net, depth(512), [3, 3], 1, data_format)
    return net

