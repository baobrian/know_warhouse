# coding=utf-8

import pandas as pd
import numpy as np
import os



def cohort_register(pk,df):
    regist=pd.DataFrame()
    data=df[pk]
    data['setl_draw_value']=1
    data['se_int_value']=data['se_int_dt_week'].map(lambda x: 1 if not pd.isnull(x) else 0)
    total_reg=data.groupby(['fir_setl_dt_week']).size()
    total_reg.name='reg_total'
    regist['total']=total_reg
    reg_temp=pd.pivot_table(data=data,values='se_int_value',\
               index='fir_setl_dt_week',columns='setl_intval',\
               aggfunc=np.sum)
    cohort_reg=reg_temp.reindex(columns=[i for i in range(0,31,1)]).fillna(0)
    cohort_30=regist.join(cohort_reg)
    return cohort_30


if __name__ == '__main__':
    path='C:\Users\Administrator\Desktop\liushi\data\\re_setl\\'
    pk=['fir_setl_dt_week','se_int_dt_week','setl_intval']
    dirs=os.listdir(path)
    result_cohort=pd.DataFrame()
    for fl in dirs:
        temp=pd.read_csv(path+fl)
        cohort_temp=cohort_register(df=temp,pk=pk)
        result_cohort=result_cohort.append(cohort_temp)
    os.chdir('C:\Users\Administrator\Desktop\liushi\\result\\')
    result_cohort.to_excel('setl_seint_draw.xlsx',header=True)




pass