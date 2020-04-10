from xmlrpc.client import boolean

import tensorflow as tf

from params.global_params import GlobalParams
from params.model_params import ModelParams


class MlpModel:

    def __init__(self, params:GlobalParams, training:bool=False):
        self.params = params
        self.training = training

    def get_model(self) -> ModelParams:
        mp = ModelParams()
        seq_length = self.params.seq_length
        data_dim = self.params.data_dim
        hidden_dims = self.params.hidden_dims
        output_dim = self.params.output_dim

        tf.reset_default_graph()
        mp.training = tf.placeholder(tf.bool, name="training")
        mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name="X")
        mp.X_close = tf.placeholder(tf.float32, [None, 1], name="X_close")
        mp.Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
        mp.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

        he_init = tf.contrib.layers.variance_scaling_initializer()

        net = tf.layers.flatten(mp.X)
        index = 1
        for n in hidden_dims:
            net = tf.layers.dense(net, units=n, activation=tf.nn.sigmoid, kernel_initializer=he_init, name="dense_" + str(index))
            net = tf.layers.dropout(net, rate = 1 - mp.output_keep_prob, training=self.training, name="dropout_" + str(index))
            index += 1

        mp.Y_pred = tf.layers.dense(net, output_dim, kernel_initializer=he_init, name="Y_pred")

        mp.loss = tf.reduce_sum(tf.square((mp.Y - mp.Y_pred) / (1 + mp.Y - mp.X_close)))
        optimizer = tf.train.AdamOptimizer(self.params.learning_rate, name="optimizer")
        mp.train = optimizer.minimize(mp.loss, name="train")

        # RMSE
        mp.targets = tf.placeholder(tf.float32, [None, output_dim], name="targets")
        mp.targets_close = tf.placeholder(tf.float32, [None, 1], name="targets_close")
        mp.predictions = tf.placeholder(tf.float32, [None, output_dim], name="predictions")
        mp.rmse = tf.sqrt(
            tf.reduce_mean(tf.square((mp.targets - mp.predictions) / (1 + mp.targets - mp.targets_close))))

        return mp