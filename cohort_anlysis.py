 # coding=utf-8
import pandas as pd
import numpy as np
import os

class CohortAnalysis() :
    def __init__(self, data, stat_time, end_time=None, intrv=None, boundary=30):
        self.data=data
        self.stat_time=stat_time
        self.end_time=end_time
        self.intrv = intrv
        self.boundary=boundary

    def analysis_cohort(self,isPercent=False):
        result = pd.DataFrame()
        if self.stat_time  is None or (self.end_time is None and self.intrv is None):
            print('请检查end_time和intrv参数')
            raise ValueError
        if self.stat_time is not None and  self.intrv is not None:
            pk=[self.stat_time,self.intrv]
            df = self.data[pk]
            df['intrv']=df[self.intrv]
        if self.stat_time is not None and self.end_time is not None and self.intrv is None:
            pk=[self.stat_time,self.end_time]
            df=self.data[pk]
            for com in pk:
                df[com]=pd.to_datetime(df[com],format='%Y/%m/%d')
            df['intrv']=(df[pk[1]]-df[pk[0]]).map(lambda x:x.days)
            df[pk[0]]=df[pk[0]].dt.date
        df['_valid_value'] = df['intrv'].map(lambda x: 1 if not pd.isnull(x) else 0)
        df['count_value'] = 1
        cohort_total = df.groupby([self.stat_time]).size()
        cohort_total.name = 'cohortanalysis'
        result['daily_sum'] = cohort_total
        temp = pd.pivot_table(data=df, values='_valid_value',\
                                  index=self.stat_time, columns='intrv',\
                                  aggfunc=np.sum)
        _result = temp.reindex(columns=[i for i in range(0, self.boundary, 1)]).fillna(0)
        # _result['Col_sum']=_result.apply(lambda x:x.sum(),axis=1)
        cohort_result = result.join(_result)
        cohort_result.loc['Row_sum'] = cohort_result.apply(lambda x: x.sum())
        if isPercent:
            for i in range(1,len(cohort_result.columns)):
                cohort_result[cohort_result.columns[i]]=cohort_result[cohort_result.columns[i]]/cohort_result['daily_sum']
            cohort_result=cohort_result.drop(['daily_sum'],axis=1)
            cohort_result.to_excel('cohort_result_percent' + str(self.boundary) + '.xlsx', header=True)
        else:
            cohort_result.to_excel('cohort_result' + str(self.boundary) + '.xlsx', header=True)
        return



if __name__ == '__main__':
    path='E:\Data_temp\\20190919\data\\'
    file_save_path='E:\Data_temp\\20190919\\result\\'
    files = os.listdir(path)
    data=pd.read_csv(path+files[0],index_col=0)
    print data.info()
    cohort=CohortAnalysis(data=data,stat_time='lst_suc_limit',end_time='last_used_time',boundary=180)
    os.chdir(file_save_path)
    cohort.analysis_cohort(isPercent=True)
    print '同期群分析文件已生成'
