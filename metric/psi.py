# coding=utf-8
from __future__ import division


# 导入模块
import data_preprocess as dap
import pandas as pd
import numpy as np
import math
import datetime
import copy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor


class ModelTrain(object):

    def __init__(self, x_train, y_train, x_test, y_test):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.rfr= RandomForestRegressor()
        self.lr = LogisticRegressionCV()
        self.rf = RandomForestClassifier()
        self.xg = XGBClassifier()
        self.gbc = GradientBoostingClassifier()
        self.prediction = pd.DataFrame()
        self.coef_corr = pd.DataFrame()
        self.result_ks = pd.DataFrame()
        self.result_psi = pd.DataFrame()


    # 随机森林回归
    def rfr_train(self):
        self.rfr=RandomForestRegressor(n_estimators=100,max_depth=10,criterion='mse')
        self.rfr.fit(self.x_train, self.y_train)


    def xg_train(self):
        # self.rfr=RandomForestRegressor(n_estimators=100,max_depth=10,criterion='mse')
        # self.rfr.fit(self.x_train, self.y_train)
        self.xg=XGBClassifier(max_depth=10, learning_rate=0.22, n_estimators=200)
        self.xg.fit(self.x_train, self.y_train)

    # 结果分析函数
    def model_access(self, x, y, mod_selection=None):

        # 评分函数
        global predict_proba

        if mod_selection == 'lr':
            predict_proba = self.lr.predict_proba(x)
        elif mod_selection == 'rf':
            predict_proba = self.rf.predict_proba(x)
        elif mod_selection == 'gbc':
            predict_proba = self.gbc.predict_proba(x)
        elif mod_selection == 'xg':
            predict_proba = self.xg.predict_proba(x)
        self.prediction = copy.deepcopy(x)
        self.prediction['_prob_'] = predict_proba[:, 1]

        # predict_proba = self.rfr.predict(x)
        # self.prediction = copy.deepcopy(x)
        # if mod_selection == 'rfr':
        #     self.prediction['_prob_'] = predict_proba
        # else:
        # self.prediction['_prob_'] = predict_proba[:, 1]
        self.prediction = self.prediction.join(y)

        # 按违约概率从大到小排序并分成10组
        self.prediction = self.prediction.sort_values(by='_prob_', ascending=False)
        self.prediction = self.prediction.reset_index()
        self.prediction['rank'] = np.floor((self.prediction.index / len(self.prediction) * 10) + 1)
        # 结果统计
        self.prediction['sum'] = 1
        rank = pd.pivot_table(self.prediction, index=['rank'], columns=y.name, values=['sum'], aggfunc=np.sum)
        print(rank)

    # PSI-训练集与验证集分级函数
    def psi_prepare(self, data_selection=None, mod_selection=None):

        if mod_selection == 0:
            self.result_psi = pd.DataFrame()
        if data_selection == 'train':
            self.result_psi['max'] = self.prediction.groupby('rank')['_prob_'].max().values
            self.result_psi['rank'] = self.prediction['rank'].unique()
            self.result_psi['rank-1'] = self.result_psi['rank'] - 1
            self.result_psi = pd.merge(self.result_psi, self.result_psi, how='outer', left_on=['rank'],
                                       right_on=['rank-1']).fillna(0)[:-1]
        print(self.result_psi)

        # 分级函数
        def rank(prob):
            #print '第1层 ：%s' %prob
            for index, row in self.result_psi.iterrows():
                #print 'index: %s' %index
                if (prob <= row['max_x']) & (prob > row['max_y']):
                    return row['rank_x']
            if prob > self.result_psi['max_x'].max():
                return self.result_psi['rank_x'].min()
            elif prob <= self.result_psi['max_y'].min():
                return self.result_psi['rank_x'].max()
        data_psi = pd.DataFrame()
        data_psi['rank'] = self.prediction['_prob_'].apply(rank)
        col = 'rank_{}'.format(data_selection)
        self.result_psi[col] = data_psi.groupby('rank').size().values
    # PSI-计算函数
    def psi(self, name=None):
        data_psi = self.result_psi[['rank_x', 'rank_train', 'rank_test']]
        data_psi.loc['total', 'rank_train'] = data_psi['rank_train'].sum()
        data_psi.loc['total', 'rank_test'] = data_psi['rank_test'].sum()
        # PSI= SUM((模型使用时段分级占比i-模型开发时段分级占比i)*LN(模型使用时段分级占比i/模型开发时段分级占比i))
        # ----------i 为评分等级
        test_i = data_psi['rank_test'] / data_psi.loc['total', 'rank_test']
        train_i = data_psi['rank_train'] / data_psi.loc['total', 'rank_train']
        data_psi['psi'] = (test_i - train_i) * np.log(test_i / train_i)
        data_psi.loc['total', 'psi'] = data_psi['psi'].sum()
        self.result_psi = data_psi
        self.result_psi = self.result_psi.rename(columns={'rank_x': '评分等级', 'rank_train': '模型开发时段',
                                                          'rank_test': '模型使用时段', 'psi': 'PSI'})
        print(self.result_psi)
        self.result_psi.to_csv(name + '.csv', index=False)


if __name__ == '__main__':

    # 开始时间
    start_time = datetime.datetime.now()
    print('开始时间为：%s\n' % start_time)
    train_df= pd.read_csv(r'/Users/baozilin/Downloads/20190701/22.csv')
    #train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\liushi\metric\data\\22.csv')
    print ('Step2 :  开始数据集预处理......')
    datapreprocess = dap.DataPreprocess()
    train_df = datapreprocess.num_replace_null(TRAIN=train_df, replace_type='nan')
    train_df = datapreprocess.replace_na(TRAIN=train_df)
    train_df, map_dict = datapreprocess.transform_categorical_alphabetically(TRAIN=train_df)
    print ('数据集预处理完成')
    target = train_df['is_appl']
    data = train_df.drop('is_appl', axis=1)
    x_train_, x_test_, y_train_, y_test_ = train_test_split(data, target, test_size=0.2)

    # 实例化类
    model = ModelTrain(x_train=x_train_, y_train=y_train_, x_test=x_test_, y_test=y_test_)

    # 随机森林
    # 训练集
    model.xg_train()    #后期可添加模型持久化读入
    model.model_access(x=x_train_, y=y_train_, mod_selection='xg')
    model.psi_prepare(data_selection='train', mod_selection=0)
    print('*' * 50 + '分割线' + '*' * 50)

    # 验证集
    model.model_access(x=x_test_, y=y_test_, mod_selection='xg')
    model.psi_prepare(data_selection='test')
    model.psi(name='lr_psi')
    print('*' * 50 + '逻辑回归训练验证完成' + '*' * 50)

    end_time = datetime.datetime.now()
    take_time = end_time - start_time
    print('结束时间为：%s\n' % end_time)
    print('耗时：%s\n' % take_time)
