import utils.load_data as load_data
import numpy as np
import tensorflow as tf
import network_utils
import misc_utils
from random import shuffle
import time
from math import log
from PIL import Image
from open_greg import get_greg

epochs = 1300000
batch_size = 8
size = (256, 256)
z_space = 64 * 2**((int(log(size[0], 2)) -2)//2)
print("Z_Space: ", z_space)
#            z_space = 256
learning_rate = .0001
loss_smoothing = 1000
model_path = './ffhq-data-with-fc-layers-and-no-bias-revised-with-noise-weights-high-lr-and-vgg/model.ckpt'
use_tanh_latent_space = False
use_vgg_loss = True
use_pixel_loss = True
add_noise = True
MLP_inputs = True
use_bias = False
alpha = 0.00#5 #smooth
beta = 10 if use_vgg_loss else 0 #vgg
gamma = 1 if use_pixel_loss else 0 #pixel


file_list = load_data.get_image_file_list('../ffhq-dataset/thumbnails128x128')
print(len(file_list))
shuffle(file_list)
total_number_of_images = len(file_list)

X = tf.placeholder(tf.float32, shape=(None, *size, 3), name='data')

latent_space = network_utils.encoder(X, z_space, use_tanh_latent_space, size)

latent_space_sum = tf.reduce_sum(latent_space)

output = network_utils.decoder(latent_space, batch_size, z_space, size, add_noise, MLP_inputs, use_bias)

dataset = tf.data.Dataset.from_tensor_slices(file_list)
dataset = dataset.map(lambda filename: tf.py_func(load_data.load_image, [filename, size], [tf.float32]))
dataset = dataset.shuffle(buffer_size = min(64, total_number_of_images))
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

training_vars = tf.trainable_variables()

smooth_loss = network_utils.variation_loss(output)
if use_vgg_loss:    
    vgg_loss = network_utils.get_vgg_loss(X, output)
    slim = tf.contrib.slim
    vgg_saver = tf.train.Saver(slim.get_model_variables())
else:
    vgg_loss = tf.constant(0, tf.float32)
pixel_loss = network_utils.get_loss(X, output)

loss = alpha * smooth_loss + beta * vgg_loss + gamma * pixel_loss

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = training_vars)

saver = tf.train.Saver()

start_time = time.time()
iterations = 0
quit_all = False
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if use_vgg_loss:
        vgg_saver.restore(sess, './vgg_19.ckpt')
        print("Loaded vgg values")
    try:
        saver.restore(sess, model_path)
        print("Restored model from ", model_path)
    except:
        print("No model found. Using random initialization.")
    
    #for op in sess.graph.get_operations():
    #    print(op.name)

    #adi = tf.get_default_graph().get_tensor_by_name('adain0/BiasAdd:0')

    for j in range(epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                a = sess.run(next_element)
                if a[0].shape[0] == batch_size:
                    #print(np.average(sess.run(tf.get_default_graph().get_tensor_by_name('data:0'), feed_dict = {X:a[0]})))
                    #print(np.average(sess.run(tf.get_default_graph().get_tensor_by_name('30/Conv2D:0'), feed_dict = {X:a[0]})))
                    #print(np.average(sess.run(latent_space, feed_dict = {X:a[0]})))
                    #print(sess.run(latent_space, feed_dict={X:a[0]}).shape)
                    l, sl, pl, vl, _, lss = sess.run([loss, smooth_loss, pixel_loss, vgg_loss, optimizer, latent_space_sum], feed_dict = {X:a[0]})
                    if iterations == 0:
                        running_loss = l
                    else:
                        running_loss = misc_utils.smooth_loss(l, loss_smoothing, iterations, running_loss)
                    print("{0:d}  {1:d}  {2:.2f}  {3:.2f}  {4:.2f}  {5:.2f}  {6:.2f}  {7:.2e}".format(j, iterations, running_loss, l, alpha*sl, gamma*pl, beta*vl, lss))
                    iterations += 1
            except tf.errors.OutOfRangeError:
                print("Epoch complete. Average epoch time: ", (time.time()-start_time)/(j+1))
                saver.save(sess, model_path)
                sess.run(iterator.initializer)
                a = sess.run(next_element)
                break
            except KeyboardInterrupt:
                ans = input("Would you like to quit? (y/n): ")
                if ans in ['yes', 'YES', 'y', 'Y', 'Yes']:
                    quit_all = True
                    break
                print("Would you like to save the images?")
                ans = input("([enter] saves in ./tmp - 'N' or 'no' does not save - Anything else specifies a directory): ")
                if ans in ['NO', 'no', 'n', 'N', 'No']:
                    save_dir = None
                else:
                    if ans == '':
                        save_dir = './trial_images'
                    else:
                        save_dir = './' + str(ans)
                #greg = np.expand_dims(get_greg(), 0)
                #load_data.display_image(greg, save_dir)
                #greg_re = sess.run(output, feed_dict = {X:(greg / 127.5) - 1.0})
                #load_data.display_image(load_data.unpreprocess_image(greg_re[0]), save_dir)
                #rowan = np.expand_dims(np.array(Image.open('rowan.jpg').resize((256,256), resample=Image.LANCZOS).convert("RGB")), 0)
                #load_data.display_image(rowan, save_dir)
                #greg_re = sess.run(output, feed_dict = {X:(rowan / 127.5) - 1.0})
                #load_data.display_image(load_data.unpreprocess_image(greg_re[0]), save_dir)
                #miles = np.expand_dims(np.array(Image.open('miles.jpg').resize((256,256), resample=Image.LANCZOS).convert("RGB")), 0)
                #load_data.display_image(miles, save_dir)
                #greg_re = sess.run(output, feed_dict = {X:(miles / 127.5) - 1.0})
                #load_data.display_image(load_data.unpreprocess_image(greg_re[0]), save_dir)
                temp_image = sess.run(output, feed_dict={X:a[0]})
                temp_variation = np.std(temp_image, axis=0)
                load_data.display_image(temp_variation*255, save_dir)
                for i in range(len(a[0])):
                    load_data.display_image(load_data.unpreprocess_image(a[0][i]), save_dir)
                    load_data.display_image(load_data.unpreprocess_image(temp_image[i]), save_dir)
        if quit_all:
            break
    ans = input("Would you like to save before quitting? (y/n): ")
    if ans in ['yes', 'YES', 'y', 'Y', 'Yes']:
        saver.save(sess, model_path)
        sess.run(iterator.initializer)
        print("Saved Successuflly")
                