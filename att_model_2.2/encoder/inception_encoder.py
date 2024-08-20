import tensorflow as tf
from .equ_inception_v3 import InceptionV3


def get_dims(net):
    ret = []
    for ind, d in enumerate(net.get_shape().dims):
        dim = d.value if d.value is not None else tf.shape(net)[ind]
        ret.append(dim)
    return ret


def encoder(input_shape, final_endpoint="mixed5a"):
    inputs = tf.keras.layers.Input(input_shape)
    net = InceptionV3(input_tensor=inputs, final_endpoint=final_endpoint)
    _, height, width, num_features = get_dims(net)# [d.value for d in net.get_shape().dims]
    # net = tf.keras.layers.Permute([2, 1, 3])(net)
    # net = tf.keras.layers.Reshape([width * height, num_features])(net)
    net = tf.transpose(net, [2, 0, 1, 3])
    net = tf.reshape(net, [width, -1, height * num_features])
    return tf.keras.Model(inputs=inputs, outputs=net, name='encoder')


def _encode_coordinates(net):
    h, w = (5, 96)  # according to max input size
    #batch_size, h, w, _ = net.shape.as_list()
    #if h is None or w is None:

    x, y = tf.meshgrid(tf.range(w), tf.range(h))
    w_loc = tf.one_hot(x, w, dtype=net.dtype)
    h_loc = tf.one_hot(y, h, dtype=net.dtype)
    loc = tf.concat([h_loc, w_loc], 2)
    loc = tf.tile(tf.expand_dims(loc, 0), [tf.shape(net)[0], 1, 1, 1])
    loc = loc[:, :tf.shape(net)[1], :tf.shape(net)[2], :]
    #loc = tf.cast(loc, dtype=net.dtype)
    return tf.concat([net, loc], 3)