# coding=utf-8

import pandas as pd
import numpy as np
import os



def cohort_register(pk,df):
    regist=pd.DataFrame()
    data=df[pk]
    data['draw_value']=1
    data['appl_value']=data['appl_dt'].map(lambda x: 1 if not pd.isnull(x) else 0)
    total_reg=data.groupby(['limit_actv_tm']).size()
    total_reg.name='reg_total'
    regist['total']=total_reg
    reg_temp=pd.pivot_table(data=data,values='appl_value',\
               index='limit_actv_tm',columns='draw_intval',\
               aggfunc=np.sum)
    cohort_reg=reg_temp.reindex(columns=[i for i in range(0,31,1)]).fillna(0)
    cohort_30=regist.join(cohort_reg)
    return cohort_30


if __name__ == '__main__':
    path='/Users/baozilin/Downloads/data/cohort_data/data/draw/data/'
    pk=['limit_actv_tm','appl_dt','draw_intval']
    dirs=os.listdir(path)
    result_cohort=pd.DataFrame()
    for fl in dirs:
        temp=pd.read_csv(path+fl)
        cohort_temp=cohort_register(df=temp,pk=pk)
        result_cohort=result_cohort.append(cohort_temp)
    os.chdir(path)
    result_cohort.to_excel('draw.xlsx',header=True)




pass