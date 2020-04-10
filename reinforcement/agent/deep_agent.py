import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys, traceback

from gym import spaces
from typing import List
from data.corp import Corp
from data.data_utils import DataUtils
from models.dqn_model import DqnModel
from reinforcement.agent.agent_params import AgentParams
from reinforcement.env.stock_env import StockEnv
from reinforcement.env.stock_env_params import StockEnvParams
from utils.date_utils import DateUtils
from visualization.invest_visualizer import InvestVisualizer


class DeepAgent:
    DISCOUNT_RATE = 0.9
    TARGET_COPY_CNT = 5

    DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(DIR, '..', '..', 'result')

    # 학습 결과의 컬럼명 정의
    RESULT_COLUMNS = ['no', 'code', 'name', 'invest_result', 'index_value', 'invest_date']
    FORCAST_COLUMNS = ['no', 'date', 'code', 'name', 'action']

    def __init__(self, action_space=None, train_model='mlp'):
        self._params:AgentParams = AgentParams()
        self._global_params:StockEnvParams = StockEnvParams(train_model=train_model)
        self._action_space:spaces.Discrete = action_space
        self._batch_indexs:list = []
        self._env:StockEnv = gym.make('StockEnv-v0')
        self._env.set_params(train_model=train_model)
        self._mainDQN:DqnModel = None
        self._targetDQN: DqnModel = None
        self._train_model = train_model

    def act(self, ob: np.ndarray) -> np.ndarray:
        #self._load_main_dqn()
        if len(ob.shape) < 3:
            return np.argmax(self._mainDQN.predict(ob[np.newaxis,:,:]))
        else:
            return np.argmax(self._mainDQN.predict(ob), 1)

    def acts(self, obs: np.ndarray) -> np.ndarray:
        #self._load_main_dqn()
        preds = self._mainDQN.predict(obs)
        return np.argmax(preds, 1)

    def _annealing_epsilon(self, episode: int, restored:bool) -> float:
        if restored :
            return 1. / (episode*10  + 1)
        else:
                return 1. / (episode+ 1)

    def trains(self, corps:pd.DataFrame=None)->(pd.DataFrame, pd.DataFrame):
        if corps is None:
            corp = Corp(self._global_params)
            corps = corp.get_eval_corps_auto()
        no = 1
        invest_date = self._global_params.invest_start_date + "~" + self._global_params.invest_end_date
        results = []
        info_data = []
        for index, corp_data in corps.iterrows():
            corp_code = corp_data['종목코드']
            corp_name = corp_data['회사명']
            invest_value, index_value, infos = self.train(corp_code=corp_code, corp_name=corp_name)
            result = [no, corp_code, corp_name, invest_value, index_value, invest_date]
            results.append(result)
            info_data.append(infos)
            print(result)
            # if no % 10 == 0:
            #     df_results = pd.DataFrame(results, columns=self.RESULT_COLUMNS)
            #     DataUtils.save_excel(df_results, self._get_result_file_path())
            no += 1
        df_results = pd.DataFrame(results, columns=self.RESULT_COLUMNS)
        DataUtils.save_excel(df_results, self._get_result_file_path())
        chart_data = None
        try:
            visualizer = InvestVisualizer(self._global_params)
            chart_data = visualizer.draw_invest_4reinforcement(info_data, corps)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)

        return df_results, chart_data

    def train_codes(self, corp_codes, train_model='rnn'):
        """하나의 세션으로 학습시키는 기본 모델 """
        corp = Corp()
        corps = corp.get_corps_for_codes(corp_codes)
        self.trains(corps)


    def train_months(self, start:str='2018.01', end:str='2018.11', invest_money:float=100000000)->None:

        train_model = self._global_params.train_model
        start_month = DateUtils.to_date(start, '%Y.%m')
        end_month = DateUtils.to_date(end, '%Y.%m')
        between = DateUtils.between_months(start_month, end_month)
        invest_months_result = []
        result_columns = ["month", "invest_money", "result_money"]
        MOCK_MONEY = 10000000
        chart_data = []
        for i in range(between + 1):
            # params.remove_session_file = True
            before_month_start = DateUtils.to_month_str(start_month, i - self._global_params.mock_period_months)
            before_month_end = DateUtils.to_month_str(start_month, i - 1)
            self._global_params.invest_start_date = before_month_start + '.01'
            self._global_params.invest_end_date = before_month_end + '.31'
            self._global_params.result_file_name = "MOCK_" + before_month_start + "-" + before_month_end
            self._global_params.invest_money = MOCK_MONEY
            corp = Corp(self._global_params)
            corps = corp.get_eval_corps_auto(self._global_params.invest_end_date)
            self._env.set_params(params=self._global_params)
            before_result, _ = self.trains(corps)
            now_month = DateUtils.to_month_str(start_month, i)
            before_result = corp.exclude_corps(before_result, now_month)
            before_result = before_result.sort_values(by='invest_result', ascending=False)
            before_result.index = range(len(before_result.index))
            corp10_codes = before_result.loc[:9, 'code']
            corp10_codes.index = range(len(corp10_codes.index))
            corp10 = corp.get_corps_for_codes(corp10_codes)
            corp10_len = len(corp10.index)

            self._global_params.invest_start_date = now_month + '.01'
            self._global_params.invest_end_date = now_month + '.31'
            self._global_params.result_file_name = "INVEST_" + now_month
            self._global_params.invest_money = invest_money / corp10_len
            self._env.set_params(params=self._global_params)
            now_result, invest_chart_data = self.trains(corp10)
            chart_data.append(invest_chart_data)
            invest_money = now_result['invest_result'].sum()
            result = [now_month, self._global_params.invest_money * corp10_len, invest_money]
            invest_months_result.append(result)
            print(result)

            df_imr = pd.DataFrame(invest_months_result, columns=result_columns)
            save_file_name = "recommend_months_" + start + "-" + end + ".xlsx"
            if "_" in train_model:
                save_file_path = os.path.join('result', 'reinforcement', train_model, self._global_params.ensemble_type, save_file_name)
            else:
                save_file_path = os.path.join('result', 'reinforcement', train_model, save_file_name)
            DataUtils.save_excel(df_imr, save_file_path)

            if len(chart_data) > 1:
                visualizer = InvestVisualizer(self._global_params)
                visualizer.draw_invest_months(chart_data, start, end)
                print()



    def _load_main_dqn(self, sess:tf.Session) -> None:
        if self._mainDQN is None or self._mainDQN.sess._closed :
            self._mainDQN = DqnModel(self._global_params, self._env.corp_code, name="main", session=sess)

    def _load_target_dqn(self, sess:tf.Session) -> None:
        if self._targetDQN is None or self._targetDQN.sess._closed :
            self._targetDQN = DqnModel(self._global_params, self._env.corp_code, name='target', session=sess)

    def _get_result_file_path(self) -> str:
        """결과를 저장할 경로"""
        return os.path.join(self.RESULT_DIR, 'reinforcement', self._global_params.train_model, self._global_params.result_file_name + '.xlsx')


    def train(self, corp_name=None, corp_code=None, is_forcast:bool=False, forcast_date:str=None) -> (float, float):

        self._env.set_corp(corp_name=corp_name, corp_code=corp_code, forcast_date=forcast_date)
        iterations = self._global_params.iterations
        loss_up_count = self._global_params.loss_up_count

        train_rmse_list = []
        test_reward_list = []
        up_count = 0
        up_count_4copy = 0
        test_best_reward = -99999

        tf.reset_default_graph()
        with tf.Session() as sess:
            self._load_main_dqn(sess)
            self._load_target_dqn(sess)
            sess.run(tf.global_variables_initializer())
            restored = self._mainDQN.restore()
            self._targetDQN.restore()

            copy_ops = self._get_copy_var_ops(dest_scope_name="target",src_scope_name="main")
            #sess.run(copy_ops)

            for episode in range(iterations[1]):
                self._env.set_observation_type('train')
                replay_buffer = self._get_replay_buffer(episode, restored)
                #print(replay_buffer[0:])
                loss = self._train_minibatch(replay_buffer)
                train_rmse_list.append(loss)
                test_final_value, _, _ = self._test()
                test_reward_list.append(test_final_value)

                #print(loss, test_final_value)
                if test_final_value > test_best_reward:
                    test_best_reward = test_final_value
                    up_count = 0
                    up_count_4copy = 0
                    self._mainDQN.save()
                else:
                    up_count+=1
                    up_count_4copy +=1

                if self.TARGET_COPY_CNT < up_count_4copy:
                    #self._mainDQN.restore()
                    sess.run(copy_ops)
                    self._targetDQN.save()
                    up_count_4copy = 0
                    #print('Copy to target')

                if loss_up_count < up_count:
                    #print('Best Reward : ', test_best_reward)
                    break

            invest_index_value = None
            invest_final_value = None
            infos = None
            if is_forcast == False:
                self._mainDQN.restore()
                invest_final_value, invest_index_value, infos= self._invest()
        return invest_final_value, invest_index_value, infos
        #print('Invest Reward: ', invest_final_value)


    def _test(self) -> (float, float, list):
        self._env.set_observation_type('test')
        return self._get_final_value()

    def _invest(self) -> (float, float, list):
        self._env.set_observation_type('invest')
        return self._get_final_value()

    def forcasts(self, corps_n_date:list):
        results = []
        no = 1
        for corp_n_date in corps_n_date:
            corp_code = corp_n_date[0].replace("A", "")
            corp_name = corp_n_date[1]
            forcast_date = corp_n_date[2]

            result = self.forcast(corp_name=corp_name, corp_code=corp_code, forcast_date=forcast_date)
            result.insert(0, no)
            print(result)
            results.append(result)
            no += 1
        df_comp_rmses = pd.DataFrame(results, columns=self.FORCAST_COLUMNS)
        DataUtils.save_excel(df_comp_rmses, self._get_result_file_path())


    def forcast(self, corp_name:str=None, corp_code:str=None, forcast_date:str=None) -> list:
        self._env.set_params(train_model=self._train_model, model_type='FORCAST')
        self.train(corp_name, corp_code, True, forcast_date)
        tf.reset_default_graph()
        with tf.Session() as sess:
            self._load_main_dqn(sess)
            sess.run(tf.global_variables_initializer())
            self._mainDQN.restore()

            date = self._env.get_forcast_date()
            ob = self._env.get_forcast_observation()
            action = int(self.act(ob))
            result = [date, corp_code, corp_name, action]
            return result


    def _get_replay_buffer(self, episode:int, restored:bool) -> np.ndarray:
        replay_buffer = []
        e = self._annealing_epsilon(episode, restored)
        done = False
        ob = self._env.reset()

        while not done:
            if np.random.rand() < e:
                action = self._env.action_space.sample()
            else:
                action = self.act(ob)
            #print("Train Observation", ob[0:10])
            next_ob, reward, done, _ = self._env.step(action)
            replay_buffer.append((ob, action, reward, next_ob, done))
            #print(ob[0], next_ob[0])
            ob = next_ob
        return  np.array(replay_buffer)

    def _get_final_value(self) -> (float, float, list):
        self._env.reset()
        obs = self._env.get_observations()
        actions = self.acts(obs)
        #print(actions)
        infos = []
        for action in actions:
            _, _, _, info =self._env.step(action)
            infos.append(info)
            #print(reward)

        return  self._env.get_final_value(), self._env.get_index_value(), infos

    def _get_copy_var_ops(self, *, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def _train_minibatch(self, replay_buffer: np.ndarray) -> float:
        buffer_len = len(replay_buffer)
        batch_size = self._global_params.batch_size

        loss = 99999
        for i in range(buffer_len//batch_size):
            batch_buffer = self._next_batch_shuffle(replay_buffer, i)
            op_arr = self._to_batch_array(batch_buffer, 0)
            action_arr = self._to_batch_array(batch_buffer, 1)
            reward_arr = self._to_batch_array(batch_buffer, 2)
            next_op_arr = self._to_batch_array(batch_buffer, 3)
            done_arr = self._to_batch_array(batch_buffer, 4)
            #print('train_minibatch', 'next_op_arr', type(batch_buffer), batch_buffer.shape, batch_buffer[4][0:5])

            X = op_arr
            y = self._mainDQN.predict(op_arr)
            next_pred = np.max(self._targetDQN.predict(next_op_arr), axis=1)
            #print(len(next_op_arr), len(pred), pred)
            Q_target = reward_arr + self.DISCOUNT_RATE * next_pred * ~done_arr
            y[np.arange(len(X)), action_arr] = Q_target
            #print(X[0:10], y[0:10])
            loss, _ = self._mainDQN.update(X, y)
        return loss


    def _to_batch_array(self, batch_buffer, buffer_index):
        #print(batch_buffer[buffer_index].shape)
        return np.array([x[buffer_index] for x in batch_buffer])

    def _to_batch_arr_reshape(self, batch_buffer, buffer_index):
        return np.array([[x[buffer_index]] for x in batch_buffer])

    def _to_batch_stack(self, batch_buffer, buffer_index):
        return np.vstack([x[buffer_index] for x in batch_buffer])

    def _next_batch(self, replay_buffer: list, batch_index:int = 0)->list:
        batch_size = self._global_params.batch_size
        buffer_cnt = len(replay_buffer)
        batch_cnt = buffer_cnt // batch_size

        start = batch_index * batch_size
        if batch_cnt == batch_index + 1:
            end = buffer_cnt
        else:
            end = start + batch_size
        return replay_buffer[start:end]

    def _next_batch_shuffle(self, replay_buffer:np.ndarray, buffer_index=0)->np.ndarray:
        batch_size = self._global_params.batch_size
        buffer_cnt = len(replay_buffer)
        batch_cnt = buffer_cnt // batch_size

        if buffer_index == 0:
            self.batch_indexs = np.arange(0, buffer_cnt)
            np.random.shuffle(self.batch_indexs)
        start = buffer_index * batch_size
        if batch_cnt == buffer_index +1:
            indexs = self.batch_indexs[start:]
        else :
            end = start + batch_size
            indexs = self.batch_indexs[start:end]
        return replay_buffer[indexs]



def main(train_model='cnn'):
    agent = DeepAgent(train_model=train_model)
    #agent.train("SK하이닉스")
    agent.trains()
    #agent.train_months()

if __name__ == "__main__":
    main()