# coding=utf-8

import pandas as pd
import numpy as np
import os




def concat_data(path):
    data=pd.DataFrame()
    os.chdir(path)
    files=os.listdir(path)
    for file in files:
        df=pd.read_csv(file)
        data=data.append(df)
    # print data['first_tx_distr_dt'].nunique()
    # print data['first_tx_distr_dt'].isnull().sum()
    return data




def cross_table(data,pk):
    data=data[pk]
    x1=data.groupby(by=['limit_appl_cnt'],as_index=False).count()
    x1.rename(columns={'limit_appl_cnt':'appl_times','id_no':'count'},inplace=True)
    return x1




if __name__ == '__main__':
    path='C:\Users\Administrator\Desktop\liushi\data\limit_appl_time\\'
    pk=['id_no','limit_appl_cnt']
    data=concat_data(path)

    data_temp=data.fillna({'limit_appl_cnt':0})
    data_temp.to_csv('limit_all.csv',index=False)
    result=cross_table(data=data_temp,pk=pk)
    result.to_excel('limit_appl_cnt.xlsx', header=True,index=False)


pass