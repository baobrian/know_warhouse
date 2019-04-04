# coding=utf-8

import pandas as pd
import numpy as np


df=pd.read_csv('C:\Users\Administrator\Desktop\data\\query-impala-231334.csv')
data=df[['id_no','reg_time','real_intval']]
data['reg_value']=1
data['real_value']=data['id_no'].map(lambda x: 1 if not pd.isnull(x) else 0)
data_reg_total=pd.pivot_table(data=data,values='reg_value',index='reg_time',columns='real_intval',aggfunc=np.sum)


pass