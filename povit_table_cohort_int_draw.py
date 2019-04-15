# coding=utf-8

import pandas as pd
import numpy as np
import os



def cohort_register(pk,df):
    regist=pd.DataFrame()
    data=df[pk]
    data['draw_value']=1
    data['re_appl_value']=data['re_appl_week'].map(lambda x: 1 if not pd.isnull(x) else 0)
    total_reg=data.groupby(['int_week']).size()
    total_reg.name='reg_total'
    regist['total']=total_reg
    reg_temp=pd.pivot_table(data=data,values='re_appl_value',\
               index='int_week',columns='int_intval',\
               aggfunc=np.sum)
    cohort_reg=reg_temp.reindex(columns=[i for i in range(0,31,1)]).fillna(0)
    cohort_30=regist.join(cohort_reg)
    return cohort_30


if __name__ == '__main__':
    path='/Users/baozilin/Downloads/data/cohort_data/data/re_draw/se_draw/'
    pk=['int_week','re_appl_week','int_intval']
    dirs=os.listdir(path)
    result_cohort=pd.DataFrame()
    for fl in dirs:
        temp=pd.read_csv(path+fl)
        cohort_temp=cohort_register(df=temp,pk=pk)
        result_cohort=result_cohort.append(cohort_temp)
    os.chdir(path)
    result_cohort.to_excel('re_draw.xlsx',header=True)




pass