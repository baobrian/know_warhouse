# coding=utf-8

import pandas as pd
import numpy as np
import os


class CohortAnalysis():
    def __init__(self,data,row,column,boundary):
        self.data=data
        self.row=row
        self.column=column
        self.boundary=boundary

    def analysis_cohort(self):
        result = pd.DataFrame()
        pk=[self.row,self.column]
        df = self.data[pk]
        df['count_value'] = 1
        df['_valid_value'] = df[self.column].map(lambda x: 1 if not pd.isnull(x) else 0)
        cohort_total = df.groupby([self.row]).size()
        cohort_total.name = 'cohortanalysis'
        result['sum'] = cohort_total
        temp = pd.pivot_table(data=df, values='_valid_value', \
                                  index=self.row, columns=self.column, \
                                  aggfunc=np.sum)
        _result = temp.reindex(columns=[i for i in range(0, self.boundary, 1)]).fillna(0)
        cohort_result = result.join(_result)
        cohort_result.loc['Row_sum'] = cohort_result.apply(lambda x: x.sum())

        cohort_result.to_excel('cohort_result'+str(self.boundary)+'.xlsx', header=True)
        return cohort_result



if __name__ == '__main__':
    path='/Users/baozilin/Downloads/data_temp/'
    files = os.listdir(path)
    data=pd.read_csv(path+files[1],index_col=0)
    print data.info()
    print data.columns
    print data['user_id'][(data['lst_suc_limit']=='2018/10/21') & (data['last_used_time']=='2018/10/30')]
    cohort=CohortAnalysis(data=data,row='lst_suc_limit',column='useddays',boundary=180)
    cohort.analysis_cohort()
    pass
