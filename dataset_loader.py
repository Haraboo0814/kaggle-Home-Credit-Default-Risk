import numpy as np
import glob
import pandas as pd
from sklearn import preprocessing
import merge_bureau
import merge_prev_app
from sklearn.decomposition import PCA


class DatasetLoader(object):
    def __init__(self, directory_path, undersample=False, pca=False):
        self._data, self._test = DatasetLoader.import_all_data(directory_path, undersample)

    @staticmethod
    def import_all_data(directory_path, undersample=False, pca=False):
        _x_train = []
        _y_train = []
        _x_test = []
        _id_test = []

        #データセット
        train = pd.read_csv(directory_path + "/application_train.csv")
        test = pd.read_csv(directory_path + "/application_test.csv")
        train_length = train.shape[0]
        test_length = test.shape[0]
        

        #補間・置換(train)
        null_sum = 0
        for col in train.columns:
            #欠損の補間
            drop_flag = False
            null_sum = train[col].isnull().sum()
            if null_sum > 0:
                if null_sum/train_length >= 0.2:
                    drop_flag = True
                else:
                    if train[col].dtype == object:
                        train[col] = train[col].fillna(train[col].mode()[0])
                    else:
                        train[col] = train[col].fillna(train[col].mean())
            #testにないラベルの置換
            if train[col].isin(["XNA", "Unknown", "Maternity leave"]).any():
                train[col] = train[col].replace("XNA",train[col].mode()[0])
                train[col] = train[col].replace("Unknown",train[col].mode()[0])
                train[col] = train[col].replace("Maternity leave",train[col].mode()[0])
            
            #マルチラベルをone-hotに
            if train[col].dtype == 'object':
                train = pd.concat([train, pd.get_dummies(train[col], prefix=col)], axis=1)
                drop_flag = True

            #不要列の削除
            if drop_flag:
                train = train.drop(col, axis=1)
        
        bureau = merge_bureau.merge_bureau()
        prevapp = merge_prev_app.merge_prev_app()

        train = train.merge(bureau, on="SK_ID_CURR", how="left", suffixes=["BR", ""])
        train = train.merge(prevapp, on="SK_ID_CURR", how="left", suffixes=["PA", ""])
        
        train = train.drop('SK_ID_CURR', axis=1)

        _y_train = train["TARGET"]
        train = train.drop('TARGET', axis=1)

        for col in train.columns:
            train[col] = train[col].fillna(0)

        #補間・置換(test)
        null_sum = 0
        for col in test.columns:
            #欠損の補間
            drop_flag = False
            null_sum = test[col].isnull().sum()
            if null_sum > 0:
                if null_sum/test_length >= 0.2:
                    drop_flag = True
                else:
                    if test[col].dtype == object:
                        test[col] = test[col].fillna(test[col].mode()[0])
                    else:
                        test[col] = test[col].fillna(test[col].mean())
            elif test[col].isin(['XNA', 'Unknown', 'Maternity leave']).any():
                test[col] = test[col].replace("XNA",test[col].mode()[0])
                test[col] = test[col].replace("Unknown",test[col].mode()[0])
                test[col] = test[col].replace("Maternity leave",test[col].mode()[0])
            
            #one-hotに
            if test[col].dtype == 'object':
                test = pd.concat([test, pd.get_dummies(test[col], prefix=col)], axis=1)
                drop_flag = True

            #不要列の削除
            if drop_flag:
                test = test.drop(col, axis=1)
        
        test = test.merge(bureau, on="SK_ID_CURR", how="left", suffixes=["BR", ""])
        test = test.merge(prevapp, on="SK_ID_CURR", how="left", suffixes=["PA", ""])
        
        for col in test.columns:
            test[col] = test[col].fillna(0)
        
        _id_test = test["SK_ID_CURR"]
        test = test.drop('SK_ID_CURR', axis=1)

        #正規化
        alldata = pd.concat([train,test])
        #alldata.to_csv("alldata.csv")
        column_scaler = preprocessing.MinMaxScaler()
        alldata[alldata.columns] = column_scaler.fit_transform(alldata[alldata.columns])
        row_scaler = preprocessing.MinMaxScaler()
        # 主成分分析
        if pca:
            pca = PCA(n_components=100)
            alldata = pd.DataFrame(pca.fit_transform(alldata))
        train = alldata[:train_length]
        test = alldata[train_length:]

        if undersample:
            #Target1を増やす
            train = pd.concat([train, _y_train],axis=1)
            train = DatasetLoader.under_sampling(train)
            _y_train = train["TARGET"]
            train = train.drop("TARGET",axis=1)

        _y_train = _y_train.values.tolist()


        train.to_csv("train.csv")
        test.to_csv("test.csv")

        print(train.shape)
        print(test.shape)

        _x_train = train.values.tolist()
        _x_test = test.values.tolist()
                
        return DataSet(_x_train, _y_train), UnknownDataSet(_x_test,_id_test)

    def load_train_test(self, train_rate=0.8, shuffle=True):
        """

        `Load datasets splited into training set and test set.
         訓練とテストに分けられたデータセットをロードします．

        Args:
            train_rate (float): Training rate.
            shuffle (bool): If true, shuffle dataset.

        Returns:
            Training Set (Dataset), Test Set (Dataset)

        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        raw_data = self._data
        if shuffle:
            raw_data = raw_data.shuffle()
        train_size = int(len(self._data.x) * train_rate)
        data_size = int(len(self._data.x))

        _train_set = raw_data.perm(0, train_size)
        _test_set = raw_data.perm(train_size, data_size)

        return _train_set, _test_set

    def load_predict_data(self):
        return self._test

    def load_all_train(self):
        return self._data

    def under_sampling(data):
        low_freq = data[data["TARGET"] == 1]
        low_freq_size = len(low_freq)
        print("low_freq_size = {}".format(low_freq_size))

        high_freq = data[data["TARGET"] == 0].index

        random_indices = np.random.choice(high_freq, low_freq_size, replace=False)

        high_freq_sample = data.loc[random_indices]
        pd.DataFrame(high_freq_sample)

        merged_data = pd.concat([high_freq_sample, low_freq], ignore_index=True)
        balanced_data = pd.DataFrame(merged_data)

        return balanced_data



class DataSet(object):

    def __init__(self, x, y):
        #print(x.shape)
        self._x = np.asarray(x)
        self._y = np.asarray(y)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def length(self):
        return len(self._x)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of value]", len(self._x))

    def shuffle(self):
        _list = list(zip(self._x, self._y))
        np.random.shuffle(_list)
        _x, _y = zip(*_list)
        return DataSet(np.asarray(_x), np.asarray(_y))

    def perm(self, start, end):
        if end > len(self._x):
            end = len(self._y)
        return DataSet(self._x[start:end], self._y[start:end])

    def __call__(self, batch_size=20, shuffle=True):
        """

        `A generator which yields a batch. The batch is shuffled as default.
         バッチを返すジェネレータです。 デフォルトでバッチはシャッフルされます。

        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.

        Yields:
            batch (ndarray[][][]): A batch data.

        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        _data = self.shuffle() if shuffle else self

        for start in range(0, self.length, batch_size):
            permed = _data.perm(start, start+batch_size)
            yield permed

class UnknownDataSet(object):

    def __init__(self, x, y):
        self._testx = np.asarray(x)
        self._id = np.asarray(y)

    @property
    def testx(self):
        return self._testx

    @property
    def testid(self):
        return self._id

    @property
    def length(self):
        return len(self._testx)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of value]", len(self._testx))




