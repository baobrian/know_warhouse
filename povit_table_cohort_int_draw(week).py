# coding=utf-8

import pandas as pd
import numpy as np
import os



def cohort_register(pk,df):
    regist=pd.DataFrame()
    data=df[pk]
    data['draw_value']=1
    data['re_appl_value']=data['intrv_week'].map(lambda x: 1 if not pd.isnull(x) else 0)
    total_reg=data.groupby(['draw_week']).size()
    total_reg.name='reg_total'
    regist['total']=total_reg
    reg_temp=pd.pivot_table(data=data,values='re_appl_value',\
               index='draw_week',columns='intrv_week',\
               aggfunc=np.sum)
    cohort_reg=reg_temp.reindex(columns=[i for i in range(0,180,1)]).fillna(0)
    cohort_30=regist.join(cohort_reg)
    return cohort_30


if __name__ == '__main__':
    path='C:\Users\Administrator\Desktop\liushi\data20190626\data\\'
    pk=['id_no','draw_week','intrv_week']
    dirs=os.listdir(path)
    result_cohort=pd.DataFrame()
    for fl in dirs:
        temp=pd.read_csv(path+fl)
        cohort_temp=cohort_register(df=temp,pk=pk)
        result_cohort=result_cohort.append(cohort_temp)
    os.chdir('C:\Users\Administrator\Desktop\liushi\data20190626\\')
    result_cohort.to_excel('re_appl_day(week).xlsx',header=True)




pass