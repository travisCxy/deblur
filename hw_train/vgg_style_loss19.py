import tensorflow as tf

''' 
# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
'''
content_layers = ['block1_pool', 'block2_pool', 'block3_pool']
style_layers = ['block1_pool', 'block2_pool', 'block3_pool']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False
    for layer in vgg.layers:
       print(layer.name)
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model




def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        outputs = self.vgg((inputs - 1) * 2)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(tf.cast(style_output,tf.float32))
                         for style_output in style_outputs]

        content_dict = {content_name: tf.cast(value,tf.float32)
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs, targets):
    style_weight = 1e-2
    content_weight = 1e2
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_targets = targets['style']
    content_targets = targets['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    return style_loss, content_loss


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    extractor = StyleContentModel()
    inputs = tf.image.decode_jpeg(tf.io.read_file("./data/demo.jpg"))
    inputs = tf.expand_dims(inputs, 0)
    inputs = tf.image.resize(inputs, (256, 256))
    inputs = tf.cast(inputs, tf.float32)
    targets = tf.image.decode_jpeg(tf.io.read_file("./data/demo_2.jpg"))
    targets = tf.expand_dims(targets, 0)
    targets = tf.image.resize(targets, (256, 256))
    targets = tf.cast(targets, tf.float32)
    tv = tf.image.total_variation(targets / 255)
    outputs = extractor(inputs)
    target_outputs = extractor(targets)
    loss = style_content_loss(outputs, target_outputs)
    print(loss)
