import numpy as np
import tensorflow as tf

from params.global_params import GlobalParams
from params.model_params import ModelParams
from reinforcement.agent.agent_params import AgentParams
from trains.learning import Learning


class DqnModel:

    def __init__(self, params: GlobalParams, corp_code:str, name:str="main", session:tf.Session=None) -> None:
        self._params:GlobalParams = params
        self._corp_code:str = corp_code
        self._name:str = name
        self._mp:ModelParams = None
        self._build_network()
        if session is None:
            self._load_session(session)
        else:
            self.sess = session

    def _load_session(self, session:tf.Session=None):
        #if self._name == 'main':
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.restore()

    def _build_network(self) -> None:
        train_model = self._params.train_model
        if train_model == 'mlp':
            self._build_network_mlp()
        elif train_model == 'rnn':
            self._build_network_rnn()
        elif train_model == 'cnn':
            self._build_network_cnn()



    def _build_network_mlp(self) -> None:

        mp = ModelParams()
        seq_length = self._params.seq_length
        data_dim = self._params.data_dim
        hidden_dims = self._params.hidden_dims
        output_dim = len(AgentParams.ACTIONS)
        learning_rate = self._params.learning_rate

        with tf.variable_scope(self._name):
            mp.training = tf.placeholder(tf.bool, name="training")
            mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name="X")
            mp.Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
            mp.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

            he_init = tf.contrib.layers.variance_scaling_initializer()

            net = tf.layers.flatten(mp.X)
            index = 1
            for n in hidden_dims:
                net = tf.layers.dense(net, units=n, activation=tf.nn.relu, kernel_initializer=he_init, name="dense_" + str(index))
                #net = tf.layers.dropout(net, rate=1 - mp.output_keep_prob, name="dropout_" + str(index))
                index += 1
            mp.Q_pred = tf.layers.dense(net, output_dim)
            mp.loss = tf.losses.mean_squared_error(mp.Y, mp.Q_pred)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            mp.train = optimizer.minimize(mp.loss)
            self._mp = mp

    def _build_network_rnn(self):

        mp = ModelParams()
        seq_length = self._params.seq_length
        data_dim = self._params.data_dim
        hidden_dims = self._params.hidden_dims
        output_dim = len(AgentParams.ACTIONS)
        learning_rate = self._params.learning_rate

        #tf.reset_default_graph()
        with tf.variable_scope(self._name):
            mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name="X")
            mp.training = tf.placeholder(tf.bool, name="training")
            mp.Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
            mp.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            xavier_init = tf.contrib.layers.xavier_initializer()

            cells = []
            i = 1
            for hidden_size in hidden_dims:
                cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, activation=tf.tanh,
                                               initializer=xavier_init, name="lstm_" + str(i))
                dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=mp.output_keep_prob)
                cells.append(dropout_cell)
                i += 1
            stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, mp.X, dtype=tf.float32)

            flat = tf.layers.flatten(outputs)
            mp.Q_pred = tf.layers.dense(flat, output_dim)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=mp.Q_pred, labels=mp.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            mp.train = optimizer.minimize(cost)

            correct_prediction = tf.equal(tf.argmax(mp.Q_pred, 1), tf.argmax(mp.Y, 1))
            mp.loss = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._mp = mp

    def _build_network_cnn(self):

        mp = ModelParams()
        seq_length = self._params.seq_length
        data_dim = self._params.data_dim
        cnn_filters = self._params.cnn_filters
        output_dim = len(AgentParams.ACTIONS)
        kernel_size = self._params.kernel_size
        pool_size = self._params.pool_size
        learning_rate = self._params.learning_rate

        #tf.reset_default_graph()
        with tf.variable_scope(self._name):
            mp.X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name="X")
            mp.training = tf.placeholder(tf.bool, name="training")
            mp.Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
            mp.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

            net = mp.X
            for filters in cnn_filters:
                for n in filters:
                    net = tf.layers.conv1d(net, filters=n, kernel_size=kernel_size, padding="valid",
                                           activation=tf.nn.relu)
                net = tf.layers.average_pooling1d(net, pool_size=pool_size, strides=pool_size, padding="same")
            flat = tf.layers.flatten(net)
            mp.Q_pred = tf.layers.dense(flat, output_dim)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=mp.Q_pred, labels=mp.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            mp.train = optimizer.minimize(cost)

            correct_prediction = tf.equal(tf.argmax(mp.Q_pred, 1), tf.argmax(mp.Y, 1))
            mp.loss = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._mp = mp

    def restore(self)->bool:
        learning = Learning(self._params)
        if learning.exist_learning_image(self._corp_code, True, self._name):
            saver = tf.train.Saver()
            saved_session_path = learning.get_session_path(self._corp_code, True, self._name)
            saver.restore(self.sess, saved_session_path)
            return True
        return False

    def save(self) -> None:
        saver = tf.train.Saver()
        learning = Learning(self._params)
        learning.save_learning_image(self.sess, saver, self._corp_code, True, self._name)

    def predict(self, ob: np.ndarray) -> np.ndarray:
        mp = self._mp
        #print('DqnModel()', 'predict', ob.shape)
        return self.sess.run(mp.Q_pred, feed_dict={mp.X: ob, mp.output_keep_prob: 1.0})

    def update(self, x_batch: np.ndarray, y_batch: np.ndarray) -> list:
        mp = self._mp
        feed = {
            mp.X: x_batch,
            mp.Y: y_batch,
            mp.output_keep_prob: self._params.dropout_keep
        }
        return self.sess.run([mp.loss, mp.train], feed)


