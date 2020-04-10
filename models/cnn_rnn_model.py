import tensorflow as tf
import numpy as np

from params.model_params import ModelParams


class CnnRnnModel:
    """학습 모델을 정의한다."""

    def __init__(self, params):
        self.params = params

    def get_model(self):
        """Stacted RNN Model을 그린다."""
        mp = ModelParams()
        seq_length = self.params.seq_length
        data_dim = self.params.data_dim
        hidden_dims = self.params.hidden_dims
        cnn_filter = self.params.cnn_filter
        #kernel_sizes = self.params.kernel_sizes
        #cnn_filters = self.params.cnn_filters
        output_dim = self.params.output_dim
        kernel_size = self.params.kernel_size
        #pool_size = self.params.pool_size

        tf.reset_default_graph()
        mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        mp.training = tf.placeholder(tf.bool)
        mp.X_close = tf.placeholder(tf.float32, [None, 1])
        mp.Y = tf.placeholder(tf.float32, [None, 1])
        mp.output_keep_prob = tf.placeholder(tf.float32)

        # pool = mp.X
        # for n in cnn_filters:
        #     conv = tf.layers.conv1d(pool, filters=n, kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
        #     pool = tf.layers.max_pooling1d(conv, pool_size=pool_size, strides=pool_size, padding="same")
        # flat_cnn = tf.layers.flatten(pool)
        # pred_cnn = tf.layers.dense(flat_cnn, 1024)
        #
        # cells = []
        # for n in hidden_dims:
        #     cell = tf.nn.rnn_cell.LSTMCell(num_units=n, activation=tf.tanh,
        #                                    initializer=tf.contrib.layers.xavier_initializer())
        #     dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=mp.output_keep_prob)
        #     cells.append(dropout_cell)
        # stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        # outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, mp.X, dtype=tf.float32)
        # flat_rnn = tf.layers.flatten(outputs)
        # pred_rnn = tf.layers.dense(flat_rnn, 1024)
        #
        # mp.Y_pred = tf.layers.dense(tf.reduce_mean([pred_cnn, pred_rnn], 0), output_dim)

        conv = tf.layers.conv1d(mp.X, filters=cnn_filter, kernel_size=kernel_size, padding="valid",
                                activation=tf.nn.relu)

        cells = []
        for n in hidden_dims:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=n, activation=tf.tanh,
                                           initializer= tf.contrib.layers.xavier_initializer())
            dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=mp.output_keep_prob)
            cells.append(dropout_cell)
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, conv, dtype=tf.float32)
        flat = tf.layers.flatten(outputs)
        mp.Y_pred = tf.layers.dense(flat, output_dim)

        tf.layers.separable_conv1d

        # pool = mp.X
        # for n in cnn_filters:
        #     conv = tf.layers.conv1d(pool, filters=n, kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
        #     pool = tf.layers.max_pooling1d(conv, pool_size=pool_size, strides=pool_size, padding="same")
        #
        # cells = []
        # for n in hidden_dims:
        #     cell = tf.nn.rnn_cell.LSTMCell(num_units=n, activation=tf.tanh,
        #                                    initializer=tf.contrib.layers.xavier_initializer())
        #     dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=mp.output_keep_prob)
        #     cells.append(dropout_cell)
        # stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        # outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, pool, dtype=tf.float32)
        # flat = tf.layers.flatten(outputs)
        # mp.Y_pred = tf.layers.dense(flat, output_dim)


        mp.loss = tf.reduce_sum(tf.square((mp.Y - mp.Y_pred) / (1 + mp.Y - mp.X_close)))
        optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
        mp.train = optimizer.minimize(mp.loss)

        # RMSE
        mp.targets = tf.placeholder(tf.float32, [None, 1])
        mp.targets_close = tf.placeholder(tf.float32, [None, 1])
        mp.predictions = tf.placeholder(tf.float32, [None, 1])
        mp.rmse = tf.sqrt(tf.reduce_mean(tf.square((mp.targets - mp.predictions) / (1 + mp.targets - mp.targets_close))))

        return mp

if __name__ == '__main__':
    a =  np.array( [[[1,2], [3,4]], [[4,5], [6,7]]])
    print(a[:,-1][:,0].reshape(-1, 1))
