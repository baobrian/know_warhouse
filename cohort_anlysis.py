# coding=utf-8

import pandas as pd
import numpy as np
import os


class CohortAnalysis():
    def __init__(self,data,stat_time,end_time,intrv,boundary):
        self.data=data
        self.stat_time=stat_time
        self.end_time=end_time
        self.intrv = intrv
        self.boundary=boundary

    def analysis_cohort(self):
        result = pd.DataFrame()
        if self.stat_time  is None or (self.end_time is None and self.intrv is None):
            raise ValueError

        if self.stat_time is not None and  (self.end_time is not None or self.intrv is not None):
            pk=[self.stat_time,self.end_time,self.intrv]
            df = self.data[pk]
            df['_valid_value'] = df[self.column].map(lambda x: 1 if not pd.isnull(x) else 0)
            pk=[self.row,self.column]
            df=self.data[pk]
            for com in pk:
                df[com]=pd.to_datetime(df[com],format='%Y/%m/%d')
            df[]=(df[pk[1]]-df[pk[0]]).map(lambda x:x.days)
            df['_valid_value'] = df['intrv'].map(lambda x: 1 if not pd.isnull(x) else 0)
        df['count_value'] = 1
        cohort_total = df.groupby([self.row]).size()
        cohort_total.name = 'cohortanalysis'
        result['daily_sum'] = cohort_total
        temp = pd.pivot_table(data=df, values='_valid_value', \
                                  index=self.row, columns='intrv', \
                                  aggfunc=np.sum)
        _result = temp.reindex(columns=[i for i in range(0, self.boundary, 1)]).fillna(0)
        # _result['Col_sum']=_result.apply(lambda x:x.sum(),axis=1)
        cohort_result = result.join(_result)
        cohort_result.loc['Row_sum'] = cohort_result.apply(lambda x: x.sum())
        cohort_result.to_excel('cohort_result1'+str(self.boundary)+'.xlsx', header=True)
        return cohort_result



if __name__ == '__main__':
    path='E:\Data_temp\\20190919\data\\'
    file_save_path='E:\Data_temp\\20190919\\result\\'
    files = os.listdir(path)
    data=pd.read_csv(path+files[0],index_col=0)
    print data.info()
    print data.columns
    print data['user_id'][(data['lst_suc_limit']=='2018/10/21') & (data['last_used_time']=='2018/10/30')]
    cohort=CohortAnalysis(data=data,row='lst_suc_limit',column='last_used_time',boundary=180)
    os.chdir(file_save_path)
    # cohort.analysis_cohort(isDetail=True)
    cohort.analysis_cohort(isDetail=True)
    pass
