from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
import pandas as pd
import os

from data.corp import Corp
from data.data_utils import DataUtils
from params.plain_model_params import PlainModelParams
from visualization.main_visualizer import MainVisualizer


class CorpsCluster:

    def __init__(self, params):
        DIR = os.path.dirname(os.path.abspath(__file__))
        self.DIR_RESULT = os.path.join(DIR, '..', 'result')
        self.params = params
        self.main = MainVisualizer(params)

    def get_corps(self):
        corp = Corp()
        return corp.get_corps3()

    def hierarchical_clustering(self):
        corps = self.get_corps()[["회사명", "시가총액\n(억)", "자사주\n(만주)", "자사주\n비중\n(%)",
                                 "18년\nOPM\n(%)", "5년평균\nPBR", "5년평균\nPER", "부채\n(억)", "자본\n(억)",
                                 "총자산\n(억)", "영업이익\n(억)", "6개월\n외인(%)", "6개월\n기관(%)"]]
        corps = corps.dropna()
        corps.set_index("회사명", inplace=True)
        plt = self.main.get_plt()
        plt.figure(figsize=(50, 10))
        mergings = linkage(corps, method='ward')
        dendrogram(mergings, labels=corps.index, leaf_rotation=90,leaf_font_size=12)
        plt.show()
        plt.close()


    def k_means(self):
        corps = self.get_corps()[["회사명", "시가총액\n(억)", "자사주\n(만주)", "자사주\n비중\n(%)",
                                 "18년\nOPM\n(%)", "5년평균\nPBR", "5년평균\nPER", "부채\n(억)", "자본\n(억)",
                                 "총자산\n(억)", "영업이익\n(억)", "6개월\n외인(%)", "6개월\n기관(%)"]]
        corps = corps.dropna()
        corps.set_index("회사명", inplace=True)

        kmeans = KMeans(n_clusters=10)
        # Fitting the input data
        kmeans.fit(corps)
        clust_labels = kmeans.predict(corps)
        #print(corps.index[clust_labels])
        # Centroid values
        #centroids = kmeans.cluster_centers_

        corps.insert(0, 'kmeans', clust_labels)
        result = corps[['kmeans']]
        print(result)
        result.insert(0, '회사명', corps.index)
        file_path = os.path.join(self.DIR_RESULT, 'K-Means.xlsx')
        DataUtils.save_excel(result, file_path)




if __name__ == '__main__':
    params = PlainModelParams()
    cluster = CorpsCluster(params)
    #corps = cluster.get_corps()
    #print(corps.columns.values)
    #cluster.hierarchical_clustering()
    cluster.k_means()

