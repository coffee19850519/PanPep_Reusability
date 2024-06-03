"""
https://blog.csdn.net/wyn1564464568/article/details/125898241
"""
import joblib
import os
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from numpy import linalg


class PCA_:
    """
    dataset : array-like of shape (n_features, n_samples)
    """

    def __init__(self, dataset):
        self.dataset = np.matrix(dataset, dtype='float64').T

    def principal_comps(self, threshold=0.90, num=7):
        # 返回满足要求的特征向量 threshold可选参数表示方差累计达到threshold后就不再取后面的特征向量
        ret = []
        data = []
        # 标准化
        for (index, line) in enumerate(self.dataset):
            self.dataset[index] -= np.mean(line)
            # self.dataset[index] /= np.std(line, ddof=1)
        # 求协方差矩阵
        Cov = np.cov(self.dataset)
        # 求特征值和特征向量
        eigs, vectors = linalg.eig(Cov)
        for i in range(len(eigs)):
            data.append((eigs[i], vectors[:, i].T))
        # 按照特征值从大到小排序
        data.sort(key=lambda x: x[0], reverse=True)
        sum = 0
        num_ = 0
        Cumulative_variance = []
        for comp in data:
            num_ += 1
            sum += comp[0] / np.sum(eigs)
            ret.append(tuple(map(lambda x: np.round(x, 5), (comp[1], comp[0] / np.sum(eigs), sum))))
            Cumulative_variance.append(ret[-1][2])
            # print('方差贡献率:', ret[-1][1], '累计方差贡献率:', ret[-1][2])
            # print('特征值:', comp[0], '特征向量:', ret[-1][0], '方差贡献率:', ret[-1][1], '累计方差贡献率:', ret[-1][2])
            if (sum > threshold) and (num_ >= num):
                print(','.join([str(n.real) for n in Cumulative_variance[: num]]))
                return ret
        return ret


if __name__ == '__main__':
    # model_csv = 'model_para.csv'
    # model_pkl_path = '/media/fei/Data/lk_code/PanPep_similar/Result_data_simi/Round1/kfold1/models.pkl'
    model_path = os.path.join("/mnt/Data1/yzy/code/PanPep_reusability/new_data/", "New_HLA-B-beta",
                              "Strategy1", "train-MA2")
    model_pkl_path = []
    for i in range(1, 6):
        model_pkl_path.append(os.path.join(model_path, "train_KFold_" + str(i), "models.pkl"))

    for i in range(len(model_pkl_path)):
        print(model_pkl_path[i])
        model_csv = "MHC2" + str(i) + ".csv"
        if not os.path.exists(model_csv):
            model_para = joblib.load(model_pkl_path[i])
            save_csv = np.array(model_para.cpu())
            pd.DataFrame(save_csv).to_csv(model_csv, index=False)

        model_p = pd.read_csv(model_csv)
        result = PCA_(model_p)
        result.principal_comps()

        # 使用sklearn
        # n_components 指明了降到几维
        pca1 = PCA(n_components=7)
        pca1.fit(model_p)
        # 打印各主成分的方差占比
        print(pca1.explained_variance_ratio_)

        print()
