# coding=utf-8

import pandas as pd
import numpy as np


df=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231334.csv')
df1=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231706.csv')
df2=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231721.csv')
df3=pd.read_csv('/Users/baozilin/Downloads/data/query-impala-231813.csv')

data=df[['id_no','reg_time','real_intval']]
data['reg_value']=1
data['real_value']=data['id_no'].map(lambda x: 1 if not pd.isnull(x) else 0)
cross_tab=data.groupby(['reg_time'])
print cross_tab.agg('count')
data_reg_total=pd.pivot_table(data=data,values='reg_value',index='reg_time',columns='real_intval',aggfunc=np.sum)
print data_reg_total.sum(axis=1)
# 你好
print (data['reg_time']=='2018-06-13').sum()
print (data['reg_time']=='2018-06-14').sum()
pass