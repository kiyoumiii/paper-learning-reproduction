import tensorflow as tf
import numpy as np
from math import log


initializer = tf.initializers.random_normal(0, 0.01)    

def encoder(net, latent_size, use_tanh, input_shape):
    conv_blocks = int(log(input_shape[0], 2))
    base_chans = int(latent_size / (2**((conv_blocks - 1)//2)))
    for j in range(conv_blocks):
        #curr_channels = min(latent_size, 32 * 2**(j//2))
        curr_channels = base_chans * 2**(j//2)
        for i in range(2):
            net = tf.layers.conv2d(net, curr_channels, 3, padding='same',
                                   kernel_initializer = initializer,
                                   activation = tf.nn.leaky_relu, name=str(j)+str(i)) 
            net = tf.layers.batch_normalization(net)                
        if j == conv_blocks - 1 and use_tanh:
            net = tf.layers.conv2d(net, curr_channels, 3, strides=(2,2), 
                                   kernel_initializer = initializer, padding='same',
                                   activation = tf.nn.tanh, name=str(j)+str(i)+'f')
        else:
            net = tf.layers.conv2d(net, curr_channels, 3, strides=(2,2), 
                                   kernel_initializer = initializer, padding='same',
                                   activation = tf.nn.leaky_relu, name=str(j)+str(i)+'m')
            net = tf.layers.batch_normalization(net)                
        
    return net

def fix_dims(t):
    return tf.expand_dims(tf.expand_dims(t, 1), 1)

def adain(x, y, c, u, i):
    #xb, xw, xh, xc = tf.shape(x)
    y = tf.reshape(tf.layers.dense(y, c*2, tf.nn.leaky_relu, kernel_initializer = initializer, use_bias = u, name='adain'+str(i)), (-1, c, 2))
    mu_x, var_x = tf.nn.moments(x, axes=[1,2], keep_dims=True)
    std_x = tf.sqrt(var_x)
    mu_y = fix_dims(y[:,:,0])
    std_y = fix_dims(y[:,:,1])
    #return std_y*(x - mu_x)/std_x + mu_y
    #top = x - mu_x
    #top = std_y  * top
    #bottom = std_x 
    #return top / bottom + mu_y
    norm_x = (x - mu_x) / std_x
    return (std_y + 1) * norm_x + mu_y
    


def decoder(latent_space, batch_size, latent_size, output_shape, add_noise, MLP_inputs, use_bias):
    conv_blocks = int(log(output_shape[0], 2)) -1
    latent_space = tf.reshape(latent_space, (-1, latent_size))
    n = 1. if add_noise else 0.
    
    if MLP_inputs:
        for _ in range(8):
            latent_space = tf.layers.dense(latent_space, latent_size, tf.nn.leaky_relu, kernel_initializer = initializer, use_bias = use_bias)

    init_val = np.random.normal(size=(4, 4, latent_size)).astype(np.float32)
    base_image = tf.Variable(init_val, dtype=tf.float32)
    dec = tf.stack([base_image] * batch_size, name='base_img')

    for i in range(conv_blocks):
        curr_channels = int(latent_size / 2 ** i)
        if i != 0:
            dec = tf.layers.conv2d_transpose(dec, curr_channels, 3, strides = (2,2), 
                                             kernel_initializer = initializer,
                                             activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)
            dec = tf.layers.conv2d(dec, curr_channels, 3, kernel_initializer = initializer, 
                                   activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)
            #dec = tf.layers.batch_normalization(dec)                
            
        weights1 = tf.Variable(np.zeros(dec.shape[1].value), dtype=tf.float32)
        dec = dec + tf.random.normal([tf.shape(dec)[0], 1, tf.shape(dec)[2], tf.shape(dec)[3]]) * tf.reshape(weights1, [1, -1, 1, 1]) * n
        dec = adain(dec, latent_space, curr_channels, use_bias, i*2)
        dec = tf.layers.conv2d(dec, curr_channels, 3, kernel_initializer = initializer, 
                               activation = tf.nn.leaky_relu, padding='same', use_bias = use_bias)
        #dec = tf.layers.batch_normalization(dec)                
        weights2 = tf.Variable(np.zeros(dec.shape[1].value), dtype=tf.float32)
        dec = dec + tf.random.normal([tf.shape(dec)[0], 1, tf.shape(dec)[2], tf.shape(dec)[3]]) * tf.reshape(weights2, [1, -1, 1, 1]) * n
        dec = adain(dec, latent_space, curr_channels, use_bias, i*2+1)
            
    output = tf.layers.conv2d(dec, 3, 1, kernel_initializer = initializer, 
                              activation = tf.nn.tanh, padding='valid', use_bias = use_bias)

    return output
    

def get_loss(x, y):
    return tf.reduce_sum(tf.pow((x - y), 2))

def get_vgg_loss(x, y):
    from tensorflow.contrib.slim.nets import vgg as model_module
    combined_images = tf.concat([x, y], axis=0)
    input_img = (combined_images + 1.0) / 2.0
    VGG_MEANS = np.array([[[[0.485, 0.456, 0.406]]]]).astype('float32')
    VGG_MEANS = tf.constant(VGG_MEANS, shape=[1,1,1,3])
    vgg_input = (input_img - VGG_MEANS) * 255.0
    bgr_input = tf.stack([vgg_input[:,:,:,2], 
                          vgg_input[:,:,:,1], 
                          vgg_input[:,:,:,0]], axis=-1)
        
    slim = tf.contrib.slim
    with slim.arg_scope(model_module.vgg_arg_scope()):
        _, end_points = model_module.vgg_19(
        bgr_input, num_classes=1000, spatial_squeeze = False, is_training=False)

    loss = 0
    for layer in ['vgg_19/conv3/conv3_1', 'vgg_19/conv5/conv5_1']:
        layer_shape = tf.shape(end_points[layer])
        x_vals = end_points[layer][:layer_shape[0]//2]
        y_vals = end_points[layer][layer_shape[0]//2:]
        loss += tf.reduce_mean(tf.pow(x_vals - y_vals, 2))

    return loss

def variation_loss(x):
    return tf.reduce_sum(tf.image.total_variation(x))
