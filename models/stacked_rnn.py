import tensorflow.compat.v1 as tf
import numpy as np

from params.model_params import ModelParams

tf.disable_v2_behavior()


class StackedRnn:
    """학습 모델을 정의한다."""

    def __init__(self, params):
        self.params = params

    def get_model(self):
        """Stacted RNN Model을 그린다."""
        mp = ModelParams()
        seq_length = self.params.seq_length
        data_dim = self.params.data_dim
        hidden_dims = self.params.hidden_dims
        output_dim = self.params.output_dim

        tf.reset_default_graph()
        mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        mp.training = tf.placeholder(tf.bool)
        mp.X_close = tf.placeholder(tf.float32, [None, 1])
        mp.Y = tf.placeholder(tf.float32, [None, output_dim])
        mp.output_keep_prob = tf.placeholder(tf.float32)

        cells = []
        for n in hidden_dims:
            #cell = tf.nn.rnn_cell.LSTMCell(num_units=n, activation=tf.tanh, initializer= tf.contrib.layers.xavier_initializer())
            cell = tf.nn.rnn_cell.LSTMCell(num_units=n, activation=tf.tanh,
                                           initializer=tf.initializers.glorot_uniform())
            dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=mp.output_keep_prob)
            cells.append(dropout_cell)
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, mp.X, dtype=tf.float32)
        mp.Y_pred = tf.layers.dense(outputs[:, -1], self.params.output_dim)
        #mp.Y_pred = tf.contrib.layers.fully_connected( outputs[:, -1], self.params.output_dim, activation_fn=None)  # We use the last cell's output
        #mp.loss = tf.reduce_sum(tf.square(mp.Y - mp.Y_pred))
        mp.loss = tf.reduce_sum(tf.square((mp.Y - mp.Y_pred) / (1 + mp.Y - mp.X_close)))

        optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
        mp.train = optimizer.minimize(mp.loss)

        # RMSE
        mp.targets = tf.placeholder(tf.float32, [None, output_dim])
        mp.targets_close = tf.placeholder(tf.float32, [None, 1])
        mp.predictions = tf.placeholder(tf.float32, [None, output_dim])
        #mp.rmse = tf.sqrt(tf.reduce_mean(tf.square(mp.targets - mp.predictions)))
        mp.rmse = tf.sqrt(tf.reduce_mean(tf.square((mp.targets - mp.predictions) / (1 + mp.targets - mp.targets_close))))

        return mp

if __name__ == '__main__':
    a =  np.array( [[[1,2], [3,4]], [[4,5], [6,7]]])
    print(a[:,-1][:,0].reshape(-1, 1))
