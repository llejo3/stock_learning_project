import tensorflow as tf
import numpy as np

from params.model_params import ModelParams


class CnnModel:

    def __init__(self, params):
        self.params = params

    def get_model(self):
        mp = ModelParams()
        seq_length = self.params.seq_length
        data_dim = self.params.data_dim
        cnn_filters = self.params.cnn_filters
        output_dim = self.params.output_dim
        kernel_size = self.params.kernel_size
        pool_size = self.params.pool_size

        tf.reset_default_graph()
        mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        mp.training = tf.placeholder(tf.bool)
        mp.X_close = tf.placeholder(tf.float32, [None, 1])
        mp.Y = tf.placeholder(tf.float32, [None, output_dim])
        mp.output_keep_prob = tf.placeholder(tf.float32)

        net = mp.X
        for filters in cnn_filters:
            for n in filters:
                net = tf.layers.conv1d(net, filters=n, kernel_size=kernel_size, padding="valid", activation=tf.nn.relu)
            net = tf.layers.average_pooling1d(net, pool_size=pool_size, strides=pool_size, padding="same")
        flat = tf.layers.flatten(net)
        mp.Y_pred = tf.layers.dense(flat, output_dim)

        mp.loss = tf.reduce_sum(tf.square((mp.Y - mp.Y_pred) / (1 + mp.Y - mp.X_close)))
        optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
        mp.train = optimizer.minimize(mp.loss)

        # RMSE
        mp.targets = tf.placeholder(tf.float32, [None, output_dim])
        mp.targets_close = tf.placeholder(tf.float32, [None, 1])
        mp.predictions = tf.placeholder(tf.float32, [None, output_dim])
        mp.rmse = tf.sqrt(
            tf.reduce_mean(tf.square((mp.targets - mp.predictions) / (1 + mp.targets - mp.targets_close))))

        return mp