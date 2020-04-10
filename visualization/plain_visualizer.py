import graphviz
from sklearn import tree
import os
import pandas as pd
import seaborn as sns

from data.data_utils import DataUtils
from visualization.main_visualizer import MainVisualizer


class PlainVisualizer:

    def __init__(self, params):
        self.params = params
        self.main = MainVisualizer(params)
        self.DIR =  os.path.join(self.main.DIR_CHARTS, 'plain')


    def draw_chart(self, model, corp_name, dts):
        if self.params.model_type == 'decision_tree':
            self.draw_tree(model, corp_name)
        elif self.params.model_type == 'linear_regression':
            self.draw_linear(dts, corp_name)
        elif self.params.model_type == 'random_forest':
            self.draw_tree(model.estimators_[5], corp_name)
        elif self.params.model_type == 'light_gbm':
            pass
            #self.draw_right_bgm(model, corp_name)
            #self.draw_random_forest(model, dts, corp_name)


    def draw_tree(self, clf,corp_name):
        file_path = os.path.join(self.DIR, self.params.model_type, corp_name)
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=['close', 'open', 'high', 'low', 'volume'],
                                        class_names=['closeY'])
        graph = graphviz.Source(dot_data)
        graph.render(file_path)

    def get_file_path(self, corp_name):
        file_path = os.path.join(self.DIR, self.params.model_type, corp_name + ".png")
        DataUtils.create_dir_4path(file_path)
        return file_path

    def draw_linear(self, dts, corp_name):
        file_path = self.get_file_path(corp_name)

        x = dts.trainX
        y = dts.trainY
        df = pd.DataFrame(x, columns=['close', 'open', 'high', 'low', 'volume'])
        df.insert(0, 'close_y', y)

        #sns_plot = sns.pairplot(data=df, diag_kind="reg")
        sns_plot = sns.pairplot(data=df, diag_kind="kde")
        sns_plot.savefig(file_path, format='png')

    def draw_light_gbm(self, dts, corp_name):
        file_path = self.get_file_path(corp_name)

        x = dts.trainX
        y = dts.trainY
        df = pd.DataFrame(x, columns=['close', 'open', 'high', 'low', 'volume'])
        df.insert(0, 'close_y', y)

        #sns_plot = sns.pairplot(data=df, diag_kind="reg")
        sns_plot = sns.pairplot(data=df, diag_kind="kde")
        sns_plot.savefig(file_path, format='png')


