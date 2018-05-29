""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk

from utils import *
from model import *
import model

##======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
import pickle
with open("./Data/_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("./Data/_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
with open("./Data/_image_test.pickle", 'rb') as f:
    _, images_test = pickle.load(f)
with open("./Data/_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("./Data/_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)
images_train = np.array(images_train)
images_test = np.array(images_test)


ni = int(np.ceil(np.sqrt(batch_size)))
save_dir = "checkpoint"


def main_train_encoder():
     """ for Style Transfer """
     generator_txt2img = model.generator_txt2img_resnet

     ## for training
     t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
     t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

     net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False)
     net_fake_image, _ = generator_txt2img(t_z,
                     net_rnn.outputs + tf.random_normal(shape=net_rnn.outputs.get_shape(), mean=0, stddev=0.02), # NOISE ON RNN
                     is_train=True, reuse=False, batch_size=batch_size)
     net_encoder = z_encoder(net_fake_image.outputs, is_train=True, reuse=False)

     ## for evaluation
     t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
     net_z = z_encoder(t_real_image, is_train=False, reuse=True)
     net_g2, _ = generator_txt2img(net_z.outputs, net_rnn.outputs, is_train=False, reuse=True, batch_size=batch_size)

     loss = tf.reduce_mean( tf.square( tf.subtract( net_encoder.outputs, t_z) ))
     e_vars = tl.layers.get_variables_with_name('z_encoder', True, True)

     lr = 0.0002
     lr_decay = 0.5      # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
     decay_every = 100   # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
     beta1 = 0.5

     with tf.variable_scope('learning_rate'):
         lr_v = tf.Variable(lr, trainable=False)

     e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=e_vars )


     ###============================ TRAINING ====================================###
     sess = tf.InteractiveSession()
     tl.layers.initialize_global_variables(sess)

     net_g_name = os.path.join(save_dir, 'net_g.npz')
     net_encoder_name = os.path.join(save_dir, 'net_encoder.npz')

     if load_and_assign_npz(sess=sess, name=net_g_name, model=net_fake_image) is False:
         raise Exception("Cannot find net_g.npz")
     load_and_assign_npz(sess=sess, name=net_encoder_name, model=net_encoder)

     sample_size = batch_size
     idexs = get_random_int(min=0, max=n_captions_train-1, number=sample_size)
     sample_sentence = captions_ids_train[idexs]
     sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')
     sample_image = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
     # print(sample_image.shape, np.min(sample_image), np.max(sample_image), image_size)
     # exit()
     sample_image = threading_data(sample_image, prepro_img, mode='translation')    # central crop first
     save_images(sample_image, [ni, ni], 'samples/step_pretrain_encoder/train__x.png')

     n_epoch = 160 * 100
     print_freq = 1
     n_batch_epoch = int(n_images_train / batch_size)

     for epoch in range(0, n_epoch+1):
         start_time = time.time()

         if epoch !=0 and (epoch % decay_every == 0):
             new_lr_decay = lr_decay ** (epoch // decay_every)
             sess.run(tf.assign(lr_v, lr * new_lr_decay))
             log = " ** new learning rate: %f" % (lr * new_lr_decay)
             print(log)
             # logging.debug(log)
         elif epoch == 0:
             log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
             print(log)

         for step in range(n_batch_epoch):
             step_time = time.time()
             ## get matched text
             idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
             b_real_caption = captions_ids_train[idexs]
             b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
             # ## get real image
             # b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
             # ## get wrong caption
             # idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
             # b_wrong_caption = captions_ids_train[idexs]
             # b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')
             # ## get wrong image
             # idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
             # b_wrong_images = images_train[idexs2]
             # ## get noise
             b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
                 # b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)

             ## update E
             errE, _ = sess.run([loss, e_optim], feed_dict={
                             t_real_caption : b_real_caption,
                             t_z : b_z})
                             # t_real_image : b_real_images,})

             print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, e_loss: %8f" \
                         % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errE))

         if (epoch + 1) % 10 == 0:
             print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
             # print(sample_image.shape, t_real_image)
             img_gen = sess.run(net_g2.outputs, feed_dict={
                                         t_real_caption : sample_sentence,
                                         t_real_image : sample_image,})
             img_gen = threading_data(img_gen, imresize, size=[64, 64], interp='bilinear')
             save_images(img_gen, [ni, ni], 'samples/step_pretrain_encoder/train_{:02d}_g(e(x))).png'.format(epoch))

         if (epoch != 0) and (epoch % 5) == 0:
             tl.files.save_npz(net_encoder.all_params, name=net_encoder_name, sess=sess)
             print("[*] Save checkpoints SUCCESS!")



if __name__=='__main__':
     main_train_encoder()
