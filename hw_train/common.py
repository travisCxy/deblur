import tensorflow as tf


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image, image_size):
    ch1 = input_image.shape[2]
    ch2 = real_image.shape[2]
    stacked_image = tf.concat([input_image, real_image], axis=2)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[image_size, image_size, ch1 + ch2])
    return cropped_image[..., :ch1], cropped_image[..., ch1:]


def random_crop_from_raw(input_jpg, real_jpg, crop_size):
    shape = tf.image.extract_jpeg_shape(input_jpg)[:2]
    limit = shape - crop_size + 1
    offset = tf.random.uniform(tf.shape(shape), dtype=tf.int32, maxval=tf.int32.max) % limit
    crop_window = [offset[0], offset[1], crop_size, crop_size]
    return tf.image.decode_and_crop_jpeg(input_jpg, crop_window), tf.image.decode_and_crop_jpeg(real_jpg, crop_window)


# normalizing the images to [-1, 1]

def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image


def denormalize(input_image):
    input_image = (input_image + 1) * 127.5
    return tf.cast(input_image, tf.uint8)


def random_jitter(input_image, real_image, image_size):
    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image, image_size)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.rot90(input_image, 1)
        real_image = tf.image.rot90(real_image, 1)
    return input_image, real_image


def random_jitter_single(input_image, image_size):
    # randomly cropping to 256 x 256 x 3
    cropped_image = tf.image.random_crop(
        input_image, size=[image_size, image_size, input_image.shape[2]])

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        cropped_image = tf.image.flip_left_right(cropped_image)

    if tf.random.uniform(()) > 0.5:
        cropped_image = tf.image.rot90(cropped_image, 1)

    return cropped_image


def colour_distance(img1, img2):
    B_1, G_1, R_1 = tf.unstack(img1, axis=2)
    B_2, G_2, R_2 = tf.unstack(img2, axis=2)
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return tf.math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))


def get_mask(img1, img2):
    dis = colour_distance(img1, img2)
    return tf.cast(dis > 20, tf.int32)


def colour_distance_2(img1, img2):
    img1 = tf.image.rgb_to_hsv(img1)
    img2 = tf.image.rgb_to_hsv(img2)
    return tf.reduce_sum(tf.abs(img1 - img2), axis=2)


def distort_color(image):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-1).
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 1]
    """

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess(source_img, target_img,random_rotate):
    # if tf.random.uniform(()) > 0.5:
    #    source_img, target_img = random_crop(source_img, target_img, crop_size - 2)
    #    source_img = tf.pad(source_img, [[1, 1], [1, 1], [0, 0]], constant_values=0)l
    #    target_img = tf.pad(target_img, [[1, 1], [1, 1], [0, 0]], constant_values=0)
    # else:
    #    source_img = tf.image.resize(source_img, (crop_size, crop_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #    target_img = tf.image.resize(target_img, (crop_size, crop_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # source_img = distort_color(source_img)
    # source_img = tf.image.resize(source_img, (crop_size, crop_size), method=tf.image.ResizeMethod.)
    # target_img = tf.image.resize(target_img, (crop_size, crop_size), method=tf.image.ResizeMethod.)
    #source_img = tf.image.random_jpeg_quality(source_img, 50, 100)
    #if tf.random.uniform(()) > 0.5:
    #    source_img = tf.image.resize(source_img,[128,128])
    #    source_img = tf.image.resize(source_img, [256, 256])
    #    source_img = tf.cast(source_img,tf.uint8)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        source_img = tf.image.flip_left_right(source_img)
        target_img = tf.image.flip_left_right(target_img)
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        source_img = tf.image.flip_up_down(source_img)
        target_img = tf.image.flip_up_down(target_img)

    if random_rotate and tf.random.uniform(()) > 0.5:
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        source_img = tf.image.rot90(source_img, k)
        target_img = tf.image.rot90(target_img, k)

    return source_img, target_img
