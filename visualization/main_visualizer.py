import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

from params.global_params import GlobalParams


class MainVisualizer:

    DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(DIR, '..', 'result')
    FIG_SIZE = (10, 6)

    def __init__(self, params:GlobalParams)-> None:
        self.params = params

        if  hasattr(self.params, 'result_child_dir') :
            self.RESULT_DIR = os.path.join(self.DIR, '..', 'result', self.params.result_child_dir)

        if "_" in params.train_model:
            self.DIR_CHARTS = os.path.join(self.RESULT_DIR, params.train_model , params.ensemble_type, 'charts')
        else:
            self.DIR_CHARTS = os.path.join(self.RESULT_DIR, params.train_model, 'charts')
    
    def get_plt(self):
        mpl.rcParams['axes.unicode_minus'] = False
        if os.name == 'nt':
            font_name = fm.FontProperties(fname=self.params.kor_font_path, size=50).get_name()
            plt.rc('font', family=font_name)
        return plt

