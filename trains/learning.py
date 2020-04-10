#import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
import shutil
import math
import numpy as np

from data.data_utils import DataUtils
from models.cnn_model import CnnModel
from models.cnn_rnn_model import CnnRnnModel
from models.mlp_model import MlpModel
from models.stacked_rnn import StackedRnn
from params.data_params import DataParams
from params.model_params import ModelParams
from params.train_params import TrainParams
from visualization.learning_visualizer import LearningVisualizer

tf.disable_v2_behavior()

class Learning:
    """학습을 시킨다"""

    DIR = os.path.dirname(os.path.abspath(__file__))
    SESSIONS_DIR = os.path.join( DIR, '..', 'data', 'files', 'sessions')  # 세션파일의 디렉토리 경로

    def __init__(self, params):
        self.params = params

    def get_session_filename(self, corp_code):
        """저장할 세션의 파일명"""
        if self.params.is_all_corps_model:
            file_name = self.params.session_file_name
        else:
            file_name = DataUtils.to_string_corp_code(corp_code)
        return file_name

    def get_session_path(self, corp_code:str, is_reinforcement:bool=False, name:str='main') -> str:
        """저장할 세션의 경로 및 파일명"""
        file_name = self.get_session_filename(corp_code)
        return os.path.join(self.get_session_dir(corp_code, is_reinforcement, name), file_name + ".ckpt")

    def get_session_dir(self, corp_code:str, is_reinforcement:bool=False, name:str='main') -> str:
        """저장할 세션의 디렉토리"""
        file_name = self.get_session_filename(corp_code)
        if is_reinforcement:
            dir = os.path.join(self.SESSIONS_DIR, 'reinforcement', name, self.params.train_model, file_name)
        else:
            dir = os.path.join(self.SESSIONS_DIR, self.params.train_model, file_name)
        DataUtils.create_dir(dir)
        return dir

    def save_learning_image(self, sess:tf.Session, saver:tf.train.Saver, comp_code:str,
                            is_reinforcement:bool=False, name:str='main') -> None:
        """학습데이터를 저장한다."""
        file_path = self.get_session_path(comp_code, is_reinforcement, name)
        saver.save(sess, file_path)

    def exist_learning_image(self, comp_code:str, is_reinforcement:bool=False, name:str='main') -> bool:
        """학습데이터가 존재하는지 여부 """
        session_path = self.get_session_path(comp_code, is_reinforcement, name)
        return os.path.isfile(session_path + '.index')

    def get_saved_session_path(self, comp_code):
        session_dir = self.get_session_dir(comp_code)
        checkpoint = tf.train.get_checkpoint_state(session_dir)
        if checkpoint is None:
            return None
        else:
            return checkpoint.model_checkpoint_path

    def delete_learning_image(self, comp_code=''):
        """학습데이터를 삭제한다. """
        session_dir = self.get_session_dir(comp_code)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir)

    def train(self, graph_params:ModelParams, corp_code:str, corp_name:str,
              data_params:DataParams) -> TrainParams:
        """학습을 시킨다."""
        X = graph_params.X
        X_close = graph_params.X_close
        Y = graph_params.Y
        training = graph_params.training
        output_keep_prob = graph_params.output_keep_prob
        train = graph_params.train
        loss = graph_params.loss
        trainX = data_params.trainX
        trainClose = data_params.trainClose
        trainY = data_params.trainY
        train_cnt = len(trainY)
        #train_len = len(trainY)
        testX = data_params.testX
        testClose = data_params.testClose
        testY = data_params.testY

        Y_pred = graph_params.Y_pred
        targets = graph_params.targets
        targets_close = graph_params.targets_close
        rmse = graph_params.rmse
        predictions = graph_params.predictions
        loss_up_count = self.params.loss_up_count
        dropout_keep = self.params.dropout_keep
        iterations = self.params.iterations
        rmse_max = self.params.rmse_max
        batch_size = self.params.batch_size
        train_batch_type = self.params.train_batch_type

        saver = tf.train.Saver()

        restored = False
        train_rmse_list = []
        tp = TrainParams()
        tp.data_params = data_params
        tp.test_rmse_list = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if self.exist_learning_image(corp_code):
                saved_session_path = self.get_session_path(corp_code)
                saver.restore(sess, saved_session_path)
                iterations[0] = 0
                restored = True

            # Training step
            less_cnt = 0
            for i in range(iterations[1]):
                if not restored or i != 0:
                    step_loss = 0
                    if train_batch_type != None:
                        for j in range(train_cnt//batch_size):
                            trainX_batch, trainY_batch, trainClose_batch = self.next_batch(trainX, trainY, trainClose, j)
                            _, step_loss = sess.run([train, loss],
                                                    feed_dict={X: trainX_batch, Y: trainY_batch, X_close: trainClose_batch,
                                                               output_keep_prob: dropout_keep, training:True})
                    else:
                        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY, X_close: testClose,
                                                                          output_keep_prob: dropout_keep, training:True})
                    train_rmse_list.append(math.sqrt(step_loss/train_cnt))
                test_predict = sess.run(Y_pred, feed_dict={X: testX, output_keep_prob: 1.0, training:False})
                test_rmse = sess.run(rmse, feed_dict={targets: testY, targets_close: testClose,  predictions: test_predict})
                tp.test_rmse_list.append(test_rmse)
                #print(testX[:,-1][:,0].reshape(-1,1))

                if not restored and iterations[0] > i:
                    continue

                if test_rmse < tp.test_rmse:
                    self.save_learning_image(sess, saver, corp_code)
                    less_cnt = 0
                    tp.train_count = i
                    tp.test_predict = test_predict
                    tp.test_rmse = test_rmse
                else:
                    less_cnt += 1

                if  less_cnt > loss_up_count and rmse_max > tp.test_rmse:
                    break

        if self.params.save_train_graph == True and (len(tp.test_rmse_list) > 0 and restored == False) :
            visualizer = LearningVisualizer(self.params)
            visualizer.draw_rmses(train_rmse_list, tp.test_rmse_list, corp_name)
        return tp

    def get_train_model(self, training=False):
        if self.params.train_model == 'mlp':
            train_model = MlpModel(self.params, training)
        elif self.params.train_model == 'cnn':
            train_model = CnnModel(self.params)
        elif self.params.train_model == 'cnn_rnn':
            train_model = CnnRnnModel(self.params)
        else:
            train_model = StackedRnn(self.params)
        return train_model.get_model()

    def learn(self, corp_code:str, corp_name:str, data_params:DataParams) -> TrainParams:
        """그래프를 그리고 학습을 시킨다."""
        if self.params.remove_session_file == True:
            self.delete_learning_image(corp_code)

        graph_params = self.get_train_model()
        return self.train(graph_params, corp_code, corp_name, data_params)

    def get_test_rmse(self, corp_code:str, data_params:DataParams) -> TrainParams:
        """그래프를 그리고 학습을 시킨다."""
        graph_params = self.get_train_model()
        Y_pred = graph_params.Y_pred
        X = graph_params.X
        training = graph_params.training
        output_keep_prob = graph_params.output_keep_prob
        targets = graph_params.targets
        targets_close = graph_params.targets_close
        rmse = graph_params.rmse
        predictions = graph_params.predictions

        testX = data_params.testX
        testClose = data_params.testClose
        testY = data_params.testY

        tp = TrainParams()
        tp.data_params = data_params

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if self.exist_learning_image(corp_code):
                saver = tf.train.Saver()
                saved_session_path = self.get_session_path(corp_code)
                saver.restore(sess, saved_session_path)

            test_predict = sess.run(Y_pred, feed_dict={X: testX, output_keep_prob: 1.0, training: False})
            tp.test_rmse = sess.run(rmse, feed_dict={targets: testY, targets_close: testClose, predictions: test_predict})
        return tp


    def next_batch(self, trainX, trainY, trainClose, batch_index=0):
        batch_type = self.params.train_batch_type
        if batch_type == 'shuffle':
            return self.next_batch_shuffle(trainX, trainY, trainClose, batch_index)
        elif batch_type == 'random':
            return self.next_batch_random(trainX, trainY, trainClose)
        else:
            return self.next_batch_sequential(trainX, trainY, trainClose, batch_index)

    def next_batch_sequential(self, trainX, trainY, trainClose, batch_index=0):
        batch_size = self.params.batch_size
        train_cnt = len(trainY)
        batch_cnt = train_cnt // batch_size

        start = batch_index * batch_size
        if batch_cnt == batch_index +1:
            end = train_cnt
        else :
            end = start + batch_size
        return trainX[start:end], trainY[start:end], trainClose[start:end]

    def next_batch_shuffle(self, trainX, trainY, trainClose, batch_index=0):
        batch_size = self.params.batch_size
        train_cnt = len(trainY)
        batch_cnt = train_cnt // batch_size

        if batch_index == 0:
            self.batch_indexs = np.arange(0, train_cnt)
            np.random.shuffle(self.batch_indexs)
        start = batch_index * batch_size
        if batch_cnt == batch_index +1:
            end = train_cnt
        else :
            end = start + batch_size
        indexs = self.batch_indexs[start:end]
        return trainX[indexs], trainY[indexs], trainClose[indexs]

    def next_batch_random(self, trainX, trainY, trainClose):
        batch_size = self.params.batch_size
        train_cnt = len(trainY)

        choices = np.random.choice(train_cnt, batch_size)
        return trainX[choices], trainY[choices], trainClose[choices]


