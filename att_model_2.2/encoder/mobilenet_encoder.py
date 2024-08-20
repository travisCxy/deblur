import tensorflow as tf
from .mobilenet import Net


def get_dims(net):
    ret = []
    for ind, d in enumerate(net.get_shape().dims):
        dim = d.value if d.value is not None else tf.shape(net)[ind]
        ret.append(dim)
    return ret


def _encode_coordinates(net):
    batch_size, h, w, _ = net.shape.as_list()
    if h is None or w is None:
        h, w = (5, 96)  # according to max input size
    x, y = tf.meshgrid(tf.range(w), tf.range(h))
    w_loc = tf.one_hot(x, w, dtype=net.dtype)
    h_loc = tf.one_hot(y, h, dtype=net.dtype)
    loc = tf.concat([h_loc, w_loc], 2)
    loc = tf.tile(tf.expand_dims(loc, 0), [tf.shape(net)[0], 1, 1, 1])
    loc = loc[:, :tf.shape(net)[1], :tf.shape(net)[2], :]
    #loc = tf.cast(loc, dtype=net.dtype)
    return tf.concat([net, loc], 3)


def coordinate_embed_encoder(input_shape, **kargs):
    inputs = tf.keras.layers.Input(input_shape)
    net = Net(input_tensor=inputs, **kargs, final_endpoint="expand_conv_layer_14")
    net = _encode_coordinates(net)
    _, height, width, num_features = get_dims(net)
    net = tf.transpose(net, [0, 2, 1, 3])
    net = tf.reshape(net, [-1, width, height * num_features])
    return tf.keras.Model(inputs=inputs, outputs=net, name='encoder')