# coding=utf-8

import pandas as pd
import numpy as np
import os


# df=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231334.csv')
# df1=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231706.csv')
# df2=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231721.csv')
# df3=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231813.csv')
def cohort_register(pk,df):
    regist=pd.DataFrame()
    data=df[pk]
    print (data['reg_time']=='2018-06-13').sum()
    print (data['reg_time'] == '2018-06-14').sum()
    data['reg_value']=1
    data['real_value']=data['id_no'].map(lambda x: 1 if not pd.isnull(x) else 0)
    total_reg=data.groupby(['reg_time']).size()
    total_reg.name='reg_total'
    regist['total']=total_reg
    reg_temp=pd.pivot_table(data=data,values='real_value',\
               index='reg_time',columns='real_intval',\
               aggfunc=np.sum)
    cohort_reg=reg_temp.reindex(columns=[i for i in range(0,31,1)]).fillna(0)
    cohort_30=regist.join(cohort_reg)
    return cohort_30


if __name__ == '__main__':
    path='/Users/baozilin/Downloads/data/cohort_data/data/regist/data/'
    pk=['id_no','reg_time','real_intval']
    os.chdir(path)
    # dirs=os.listdir(path)
    dirs=['/Users/baozilin/Downloads/data/cohort_data/data/regist/data/query-impala-237144_06.csv']
    result_cohort=pd.DataFrame()
    for fl in dirs:
        temp=pd.read_csv(fl)
        cohort_temp=cohort_register(df=temp,pk=pk)
        result_cohort=result_cohort.append(cohort_temp)
    result_cohort.to_excel('regist.xlsx')




pass