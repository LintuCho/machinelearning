from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np



class Autocoder:

    def __init__(self, lr, num_steps, batch_size, lnn):

        self.learning_rate = lr
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.display_step = 1000
        self.examples_to_show = 10

    # Network Parameters
        self.num_hidden_1 = lnn[1]
        self.num_hidden_2 = lnn[2] 
        self.num_input = lnn[0] 

        self.X = tf.placeholder("float", [None, self.num_input])

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }

        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)


        self.y_pred = self.decoder_op

        self.y_true = self.X


        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train(self, data):
        n_sample = data.shape[0]
        sample_len = data.shape[1]
        n_batches = int(n_sample/self.batch_size)
        for i in range(1, self.num_steps+1):

            for batchIdx in range(n_batches):
                data_batch = data[batchIdx*self.batch_size:(batchIdx+1)*self.batch_size]
                _,self.l = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: data_batch})

                if i % self.display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, self.l))

        g,d = self.sess.run([self.encoder_op,self.decoder_op], feed_dict={self.X: data})
        return [g,d]


    def encoder(self,x):

        layer_1 = tf.nn.elu(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                    self.biases['encoder_b1']))

        layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                    self.biases['encoder_b2']))
        return layer_2



    def decoder(self,x):

        layer_1 = tf.nn.elu(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                    self.biases['decoder_b1']))

        layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                    self.biases['decoder_b2']))
        return layer_2