import tensorflow as tf
from absl import flags
from classification_models.tfkeras import Classifiers
import efficientnet_v2


def upsample(filters, size, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def Generator(output_channels):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs
    base_model = tf.keras.applications.MobileNetV2(input_shape=[None, None, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 5, strides=2,
        padding='same', name="logits")  # 64x64 -> 128x128

    logits = last(x)
    return tf.keras.Model(inputs=inputs, outputs=logits)



def Generator_resnet(output_channels):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs
    ResNet34, preprocess_input = Classifiers.get('resnet34')
    base_model = ResNet34((None, None, 3), weights='imagenet')
    #base_model = resnet.ResNet34((None, None, 3))
    base_model.summary()
    #base_model = tf.keras.applications.MobileNetV2(input_shape=[None, None, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'relu0',  # 64x64
        'stage1_unit3_relu2',  # 32x32
        'stage2_unit4_relu2',  # 16x16
        'stage3_unit6_relu2',  # 8x8
        'stage4_unit2_relu2',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 5, strides=2,
        padding='same', name="logits")  # 64x64 -> 128x128

    logits = last(x)
    return tf.keras.Model(inputs=inputs, outputs=logits)


def Generator_efficientnetv2(output_channels):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs
    base_model = efficientnet_v2.EfficientNetV2S(include_top=False, weights='efficientnetv2-s_notop.h5',
                                                                        input_shape=(None, None, 3))

    # base_model = efficientnet_v2.EfficientNetV2M(include_top=False, weights='efficientnetv2-m-21k-ft1k_notop.h5',
    #                                              input_shape=(None, None, 3))
    base_model.summary()

    # Use the activations of these layers
    layer_names = [
        'block1a_project_bn',  # 64x64
        'block2b_drop',  # 32x32
        'block3b_drop',  # 16x16
        'block5e_drop',  # 8x8
        'block6h_project_bn',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    up_stack = [
        upsample(192, 3),  # 4x4 -> 8x8
        upsample(112, 3),  # 8x8 -> 16x16
        upsample(48, 3),  # 16x16 -> 32x32
        upsample(32, 3),  # 32x32 -> 64x64
    ]
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 5, strides=2,
        padding='same', name="logits")  # 64x64 -> 128x128

    logits = last(x)
    return tf.keras.Model(inputs=inputs, outputs=logits)


LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    # l1_loss = tf.reduce_mean(tf.losses.mse(target, gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    image_size = None
    inp = tf.keras.layers.Input(shape=[image_size, image_size, flags.FLAGS.output_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[image_size, image_size, flags.FLAGS.output_channels], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=2,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss




if __name__ == "__main__":
    import os
    import cv2

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = Generator_resnet(3)
    tf.keras.utils.plot_model(model, to_file="/tmp/graph.png", show_shapes=True)
    img = cv2.imread(("/tmp/graph.png"))
    cv2.imshow("img", img)
    cv2.waitKey()
