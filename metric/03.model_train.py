# test model
# 2018-08
# by sky


# 导入模块
import pandas as pd
import numpy as np
import math
import datetime
import copy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


class ModelTrain(object):

    def __init__(self, x_train, y_train, x_test, y_test):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lr = LogisticRegressionCV()
        self.rf = RandomForestClassifier()
        self.gbc = GradientBoostingClassifier()
        self.prediction = pd.DataFrame()
        self.coef_corr = pd.DataFrame()
        self.result_ks = pd.DataFrame()
        self.result_psi = pd.DataFrame()

    # 模型训练函数

    # 逻辑回归
    def lr_train(self):

        # 调参
        params = [
            {'penalty': ['l1'], 'solver': ['liblinear'], 'multi_class': ['ovr']},
            {'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'sag'], 'multi_class': ['ovr', 'multinomial']}
        ]
        result = {}
        for param in params:
            grid = GridSearchCV(self.lr, param, cv=5, scoring='roc_auc')
            grid.fit(self.x_train, self.y_train)
            print(grid.best_params_, grid.best_score_)
            result[grid.best_score_] = grid.best_params_

        # 调参后再次训练
        best_params = result[max(result.keys())]
        print(best_params)
        self.lr = LogisticRegressionCV(multi_class=best_params['multi_class'],
                                       penalty=best_params['penalty'], solver=best_params['solver'])
        # pipeline_lr = PMMLPipeline([("classifier", self.lr)])
        # pipeline_lr.fit(self.x_train, self.y_train)
        # sklearn2pmml(pipeline_lr, "LR.pmml", with_repr=True)
        self.lr.fit(self.x_train, self.y_train)
        print(self.lr)

    # 随机森林
    def rf_train(self):

        # 调参
        params = [{'n_estimators': np.arange(10, 101, 10)},  # 迭代次数
                  {'max_depth': np.arange(1, 7, 1)},  # 最大树深度
                  {'min_samples_split': np.arange(10, 201, 10)},  # 内部节点再划分所需最小样本数
                  {'min_samples_leaf': np.arange(5, 41, 5)},  # 叶子节点最少样本数
                  {'max_features': np.arange(1, 10)}  # 最大特征数
                  ]
        best_params = {}
        grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params[0], scoring='roc_auc', cv=5)
        grid.fit(self.x_train, self.y_train)
        best_params.update(grid.best_params_)
        grid = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_params['n_estimators']),
                            param_grid=params[1], scoring='roc_auc', cv=5)
        grid.fit(self.x_train, self.y_train)
        best_params.update(grid.best_params_)
        grid = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                             max_depth=best_params['max_depth']),
                            param_grid=params[2], scoring='roc_auc', cv=5)
        grid.fit(self.x_train, self.y_train)
        best_params.update(grid.best_params_)
        grid = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                             max_depth=best_params['max_depth'],
                                                             min_samples_split=best_params['min_samples_split']),
                            param_grid=params[3], scoring='roc_auc', cv=5)
        grid.fit(self.x_train, self.y_train)
        best_params.update(grid.best_params_)
        grid = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                             max_depth=best_params['max_depth'],
                                                             min_samples_split=best_params['min_samples_split'],
                                                             min_samples_leaf=best_params['min_samples_leaf']),
                            param_grid=params[4], scoring='roc_auc', cv=5)
        grid.fit(self.x_train, self.y_train)
        best_params.update(grid.best_params_)
        print(best_params)
        self.rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                         max_depth=best_params['max_depth'],
                                         min_samples_split=best_params['min_samples_split'],
                                         min_samples_leaf=best_params['min_samples_leaf'],
                                         max_features=best_params['max_features'])
        # pipeline_rf = PMMLPipeline([("classifier", self.rf)])
        # pipeline_rf.fit(self.x_train, self.y_train)
        # sklearn2pmml(pipeline_rf, "RF.pmml", with_repr=True)
        self.rf.fit(self.x_train, self.y_train)
        print(self.rf)
        y_pred = self.rf.predict(self.x_train)
        y_predprob = self.rf.predict_proba(self.x_train)[:, 1]
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))
        rf_feature_importance = pd.DataFrame({'VARIATE': self.x_train.columns,
                                              'IMPORTANCE': self.rf.feature_importances_})[['VARIATE', 'IMPORTANCE']]
        rf_feature_importance = rf_feature_importance.sort_values(by='IMPORTANCE',
                                                                  ascending=False).reset_index(drop=True)
        print(rf_feature_importance)
        rf_feature_importance.to_csv('rf_feature_importance.csv', index=False)

    # GBDT函数
    def gbc_train(self):

        params = [{'n_estimators': np.arange(100, 1501, 100)},  # 调整迭代次数
                  {'learning_rate': [0.01, 0.05, 0.1]},  # 学习率
                  {'max_depth': np.arange(3, 7, 1)},  # 最大深度
                  {'min_samples_split': np.arange(5, 31, 10)},  # 内部节点再划分所需最小样本数
                  {'min_samples_leaf': np.arange(40, 71, 10)},  # 叶子节点最少样本数
                  {'max_features': np.arange(1, 15, 1)},  # 最大特征数
                  {'subsample': [0.9, 0.95]}  # 子采样比率
                  ]
        best_params = {}

        # 迭代次数
        grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=params[0], scoring='roc_auc', cv=5)
        grid.fit(self.x_train, self.y_train)
        y_pred = grid.predict(self.x_train)
        y_predprob = grid.predict_proba(self.x_train)[:, 1]
        best_params.update(grid.best_params_)
        print(grid.best_params_, grid.best_score_)
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))

        # 最大深度与内部节点再划分所需最小样本数
        params[2].update(params[3])
        grid = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=best_params['n_estimators']),
                            param_grid=params[2], scoring='roc_auc', iid=False, cv=5)
        grid.fit(self.x_train, self.y_train)
        y_pred = grid.predict(self.x_train)
        y_predprob = grid.predict_proba(self.x_train)[:, 1]
        best_params.update(grid.best_params_)
        print(grid.best_params_, grid.best_score_)
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))

        # 内部节点再划分所需最小样本数与叶子节点最少样本数
        params[3].update(params[4])
        grid = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                                                 max_depth=best_params['max_depth'],
                                                                 min_samples_split=best_params['min_samples_split']),
                            param_grid=params[3], scoring='roc_auc', iid=False, cv=5)
        grid.fit(self.x_train, self.y_train)
        y_pred = grid.predict(self.x_train)
        y_predprob = grid.predict_proba(self.x_train)[:, 1]
        best_params.update(grid.best_params_)
        print(grid.best_params_, grid.best_score_)
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))

        # 最大特征数
        grid = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                                                 max_depth=best_params['max_depth'],
                                                                 min_samples_split=best_params['min_samples_split'],
                                                                 min_samples_leaf=best_params['min_samples_leaf']
                                                                 ),
                            param_grid=params[5], scoring='roc_auc', iid=False, cv=5)
        grid.fit(self.x_train, self.y_train)
        y_pred = grid.predict(self.x_train)
        y_predprob = grid.predict_proba(self.x_train)[:, 1]
        best_params.update(grid.best_params_)
        print(grid.best_params_, grid.best_score_)
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))

        # 子采样比率
        grid = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                                                 max_depth=best_params['max_depth'],
                                                                 min_samples_split=best_params['min_samples_split'],
                                                                 min_samples_leaf=best_params['min_samples_leaf'],
                                                                 max_features=best_params['max_features']
                                                                 ),
                            param_grid=params[6], scoring='roc_auc', iid=False, cv=5)
        grid.fit(self.x_train, self.y_train)
        y_pred = grid.predict(self.x_train)
        y_predprob = grid.predict_proba(self.x_train)[:, 1]
        best_params.update(grid.best_params_)
        print(grid.best_params_, grid.best_score_)
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))

        # 学习率与迭代次数
        params[0].update(params[1])
        grid = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                                                 max_depth=best_params['max_depth'],
                                                                 min_samples_split=best_params['min_samples_split'],
                                                                 min_samples_leaf=best_params['min_samples_leaf'],
                                                                 max_features=best_params['max_features'],
                                                                 subsample=best_params['subsample']
                                                                 ),
                            param_grid=params[0], scoring='roc_auc', iid=False, cv=5)
        grid.fit(self.x_train, self.y_train)
        y_pred = grid.predict(self.x_train)
        y_predprob = grid.predict_proba(self.x_train)[:, 1]
        best_params.update(grid.best_params_)
        print(grid.best_params_, grid.best_score_)
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print('AUC Score (Train): %f' % roc_auc_score(self.y_train, y_predprob))

        # 以最优参数训练
        self.gbc = GradientBoostingClassifier(learning_rate=best_params['learning_rate'],
                                              n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              min_samples_split=best_params['min_samples_split'],
                                              max_features=best_params['max_features'],
                                              subsample=best_params['subsample'],
                                              min_samples_leaf=best_params['min_samples_leaf'])
        # pipeline_gbc = PMMLPipeline([("classifier", self.gbc)])
        # pipeline_gbc.fit(self.x_train, self.y_train)
        # sklearn2pmml(pipeline_gbc, "GBC.pmml", with_repr=True)
        self.gbc.fit(self.x_train, self.y_train)

        y_pred = self.gbc.predict(self.x_train)
        y_predprob = self.gbc.predict_proba(self.x_train)[:, 1]
        print('Accuracy : %.4g' % accuracy_score(self.y_train.values, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(self.y_train, y_predprob))
        gbc_feature_importance = pd.DataFrame({'VARIATE': self.x_train.columns,
                                               'IMPORTANCE': self.gbc.feature_importances_})[['VARIATE', 'IMPORTANCE']]
        gbc_feature_importance = gbc_feature_importance.sort_values(by='IMPORTANCE',
                                                                    ascending=False).reset_index(drop=True)
        print(gbc_feature_importance)
        gbc_feature_importance.to_csv('gbc_feature_importance.csv', index=False)

    # 逻辑回归
    def lr_coef_corr(self, var_decsribe=None, name=None):

        # 计算系数
        coef = pd.DataFrame()
        coef['VARIATE'] = self.x_train.columns
        coef['COEF'] = self.lr.coef_[0]
        coef = coef.sort_values('COEF', ascending=False)
        # 计算截距
        coef = coef.append(pd.DataFrame([['intercept', self.lr.intercept_[0]]], columns=['VARIATE', 'COEF']))
        coef = coef.reset_index(drop=True)
        # 解释变量与响应变量的相关性
        x_data = self.x_train.append(self.x_test)
        y_data = self.y_train.append(self.y_test)
        lr_corr = pd.DataFrame()
        for i in coef['VARIATE'].values:
            if i == 'intercept':
                continue
            corr = x_data[i].corr(y_data)
            lr_corr = lr_corr.append([[i, corr]])
        lr_corr.columns = ['VARIATE', 'CORR']
        lr_corr = lr_corr.sort_values('CORR', ascending=False)
        lr_corr = lr_corr.reset_index(drop=True)
        self.coef_corr = pd.merge(coef, lr_corr, on='VARIATE', how='outer')
        # 计算STD_COEF = 变量系数*变量标准差/（Π/√3）
        std = self.x_train.describe().loc['std'].reset_index().rename(columns={'std': 'STD_COEF',
                                                                               'index': 'VARIATE'})
        self.coef_corr = pd.merge(self.coef_corr, std, how='left', on='VARIATE')
        self.coef_corr['STD_COEF'] = self.coef_corr['STD_COEF'] * self.coef_corr['COEF'] / (np.pi / np.sqrt(3))
        # 计算VIF
        shape = self.x_train.shape[1]
        vif_df = pd.DataFrame()
        for i in range(shape):
            vif = variance_inflation_factor(self.x_train.as_matrix(), i)
            vif_df = vif_df.append([[self.x_train.columns[i], vif]])
        vif_df.columns = ['VARIATE', 'VIF']
        self.coef_corr = pd.merge(self.coef_corr, vif_df, how='left', on='VARIATE')
        # COEF和COEE一致性
        self.coef_corr['CONSISTENT'] = 'False'
        idx = self.coef_corr[((self.coef_corr['COEF'] >= 0) & (self.coef_corr['CORR'] >= 0)) |
                             ((self.coef_corr['COEF'] <= 0) & (self.coef_corr['CORR'] <= 0))].index
        self.coef_corr.loc[idx, 'CONSISTENT'] = 'True'
        self.coef_corr['CONSISTENT'][self.coef_corr['VARIATE'] == 'intercept'] = np.nan
        # 添加字段描述
        self.coef_corr = pd.merge(self.coef_corr, var_decsribe, how='left', on='VARIATE')
        self.coef_corr = self.coef_corr[['VARIATE', 'DESCRIPTION', 'COEF',
                                         'STD_COEF', 'VIF', 'CORR', 'CONSISTENT']]
        # 输出结果
        self.coef_corr.to_csv(name + '.csv', index=False)

    # 结果分析函数
    def model_access(self, x, y, mod_selection=None):

        # 评分函数
        global predict_proba

        # def score_calc(z):
        #     score = 500 - (20 / math.log(2)) * (math.log(15) + math.log(z / (1 - z)))
        #     return int(score)

        if mod_selection == 'lr':
            predict_proba = self.lr.predict_proba(x)
        elif mod_selection == 'rf':
            predict_proba = self.rf.predict_proba(x)
        elif mod_selection == 'gbc':
            predict_proba = self.gbc.predict_proba(x)
        self.prediction = copy.deepcopy(x)
        self.prediction['_prob_'] = predict_proba[:, 1]
        # self.prediction['score'] = self.prediction['_prob_'].apply(score_calc)
        self.prediction = self.prediction.join(y)
        # 按违约概率从大到小排序并分成10组
        self.prediction = self.prediction.sort_values(by='_prob_', ascending=False)
        self.prediction = self.prediction.reset_index()
        self.prediction['rank'] = np.floor((self.prediction.index / len(self.prediction) * 10) + 1)
        # self.prediction.to_csv('')
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

            for index, row in self.result_psi.iterrows():
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

    # KS计算函数,模型评估表
    def ks(self, good=0, bad=1, name=None):

        # #Total
        self.result_ks['# Total'] = self.prediction.groupby('rank').sum()['sum']
        # self.result_ks.loc['Total', '# Total'] = self.result_ks['# Total'][:10].sum()
        # _prob_
        self.result_ks['Min _prob_'] = self.prediction.groupby('rank')['_prob_'].min()
        # self.result_ks.loc['Total', 'Min _prob_'] = self.result_ks['Min _prob_'][:10].min()
        self.result_ks['Max _prob_'] = self.prediction.groupby('rank')['_prob_'].max()
        # self.result_ks.loc['Total', 'Max _prob_'] = self.result_ks['Max _prob_'][:10].max()
        self.result_ks['Mean _prob_'] = self.prediction.groupby('rank')['_prob_'].mean()
        # self.result_ks.loc['Total', 'Mean _prob_'] = (self.result_ks['# Total']*self.result_ks['Mean _prob_']).sum()
        #  / (self.result_ks['Mean _prob_'].sum())
        # 统计每个区间Bad 与 Non-Bad 累计占比
        col = self.y_train.name
        self.result_ks['# Non-Bad'] = self.prediction[self.prediction[col] == good].groupby('rank').sum()['sum']
        self.result_ks['% Total Non-Bad'] = self.result_ks['# Non-Bad'] / self.result_ks['# Non-Bad'].sum()
        self.result_ks['Cum %Total Non-Bad'] = self.result_ks['# Non-Bad'].cumsum() / self.result_ks['# Non-Bad'].sum()
        # 统计每个区间累计Bad 与 Non-Bad样本数占总Bad 与 Non-Bad样本数比率
        self.result_ks['# Bads'] = self.prediction[self.prediction[col] == bad].groupby('rank').sum()['sum']
        # 防止出现Nan的情况影响后续计算，对Nan替换为0
        self.result_ks.replace(np.nan, 0, inplace=True)
        self.result_ks['% Total Bads'] = self.result_ks['# Bads'] / self.result_ks['# Bads'].sum()
        self.result_ks['Cum %Total Bads'] = self.result_ks['# Bads'].cumsum() / self.result_ks['# Bads'].sum()
        self.result_ks['Interval Bads Rate'] = self.result_ks['# Bads'] / self.result_ks['# Total']
        # self.result_ks.loc['Total', 'Interval Bads Rate'] = self.result_ks['# Bads'].sum()\
        #     / self.result_ks.loc['Total', '# Total']
        # 取每个区间差值最大的为ks值
        self.result_ks['Bads vs. Non-Bad K-S'] = self.result_ks['Cum %Total Bads'] - self.result_ks[
            'Cum %Total Non-Bad']
        # self.result_ks.loc['Total', '# Non-Bad'] = self.result_ks['# Non-Bad'][:10].sum()
        # self.result_ks.loc['Total', '% Total Non-Bad'] = self.result_ks['% Total Non-Bad'][:10].sum()
        # self.result_ks.loc['Total', '# Bads'] = self.result_ks['# Bads'][:10].sum()
        # self.result_ks.loc['Total', '% Total Bads'] = self.result_ks['% Total Bads'][:10].sum()
        self.result_ks.loc['Total', 'Bads vs. Non-Bad K-S'] = self.result_ks['Bads vs. Non-Bad K-S'][:10].max()
        print('模型评估表', self.result_ks)

        # 模型评估表
        self.result_ks.to_csv(name + '.csv')

    # 绘制ROC
    def roc_curve(self, title=None, name=None):

        # 防止中文乱码
        mpl.rcParams['font.sans-serif'] = [u'simHei']
        mpl.rcParams['axes.unicode_minus'] = False
        x = range(11)
        y1 = [0] + list(self.result_ks['Cum %Total Bads'][:10].values)
        y2 = [0] + list(self.result_ks['Cum %Total Non-Bad'][:10].values)
        ks = self.result_ks[:10]['Bads vs. Non-Bad K-S'].max()
        df = self.result_ks[self.result_ks['Bads vs. Non-Bad K-S'] == ks]
        x3 = [df.index[0], df.index[0]]
        y3 = [df['Cum %Total Bads'].values[0], df['Cum %Total Non-Bad'].values[0]]
        plt.plot(x, y1, label='Cum %Total Bads', linewidth=2, color='b')
        plt.plot(x, y2, label='Cum %Total Non-Bad', linewidth=2, color='g')
        plt.plot(x3, y3, label='KS-{:.2f}'.format(ks), color='r', marker='o', markerfacecolor='r', markersize=5)
        plt.scatter(x3, y3, color='r')
        plt.title('{}'.format(title))
        plt.legend()
        plt.savefig(name + '.jpg')
        plt.show()


if __name__ == '__main__':

    # 开始时间
    start_time = datetime.datetime.now()
    print('开始时间为：%s\n' % start_time)

    # 读取数据
    data_path = r'C:\Users\Baiwei\Desktop\04.数据模型\18.中原消费\培训模型_0805'
    var_desc = pd.read_excel(data_path + '\\var_desc.xlsx')
    data_woe_path = r'C:\Users\Baiwei\Desktop\04.数据模型\18.中原消费\培训模型_0805'
    f_data_woe = open(data_woe_path + '\\data_woe.csv')
    data_woe = pd.read_csv(f_data_woe, header=0)

    # 划分训练集与验证集
    f_var_select = open(data_woe_path + '\\var_iv_select.csv')
    var_select = pd.read_csv(f_var_select)
    var = var_select[var_select['KEEP/DROP'] == 'keep']['VARIATE']
    arg_target_var = 'label_real'
    x_train_, x_test_, y_train_, y_test_ = train_test_split(data_woe[var], data_woe[arg_target_var], train_size=0.66)
    y_train_ = y_train_.apply(lambda x: int(x))
    y_test_ = y_test_.apply(lambda x: int(x))

    # 实例化类
    model = ModelTrain(x_train=x_train_, y_train=y_train_, x_test=x_test_, y_test=y_test_)
    # 逻辑回归
    # 训练集
    model.lr_train()
    model.model_access(x=x_train_, y=y_train_, mod_selection='lr')
    model.lr_coef_corr(var_decsribe=var_desc, name='lr_coef_corr')
    model.ks(name='lr_train_ks')
    model.roc_curve(title='lr_train', name='lr_train_roc')
    model.psi_prepare(data_selection='train', mod_selection=0)
    print('*' * 50 + '分割线' + '*' * 50)
    # 验证集
    model.model_access(x=x_test_, y=y_test_, mod_selection='lr')
    model.psi_prepare(data_selection='test')
    model.psi(name='lr_psi')
    model.ks(name='lr_test_ks')
    model.roc_curve(title='lr_test', name='lr_test_roc')
    print('*' * 50 + '逻辑回归训练验证完成' + '*' * 50)

    # # 随机森林
    # # 训练集
    # model.rf_train()
    # model.model_access(x=x_train_, y=y_train_, mod_selection='rf')
    # model.ks(name='rf_train_ks')
    # model.psi_prepare(data_selection='train', mod_selection=0)
    # model.roc_curve(title='rf_train', name='rf_train_roc')
    # print('*' * 50 + '分割线' + '*' * 50)
    # # 验证集
    # model.model_access(x=x_test_, y=y_test_, mod_selection='rf')
    # model.psi_prepare(data_selection='test')
    # model.psi(name='rf_psi')
    # model.ks(name='rf_test_ks')
    # model.roc_curve(title='rf_test', name='rf_test_roc')
    # print('*' * 50 + '随机森林训练验证完成' + '*' * 50)
    #
    # # GBDT
    # # 训练集
    # model.gbc_train()
    # model.model_access(x=x_train_, y=y_train_, mod_selection='gbc')
    # model.ks(name='gbc_train_ks')
    # model.roc_curve(title='gbc_train', name='gbc_train_roc')
    # model.psi_prepare(data_selection='train', mod_selection=0)
    # print('*' * 50 + '分割线' + '*' * 50)
    # # 验证集
    # model.model_access(x=x_test_, y=y_test_, mod_selection='gbc')
    # model.psi_prepare(data_selection='test')
    # model.psi(name='gbc_psi')
    # model.ks(name='gbc_test_ks')
    # model.roc_curve(title='gbc_test', name='gbc_test_roc')
    # print('*' * 50 + 'GBDT训练验证完成' + '*' * 50)

    # 结束时间及耗时
    end_time = datetime.datetime.now()
    take_time = end_time - start_time
    print('结束时间为：%s\n' % end_time)
    print('耗时：%s\n' % take_time)
