# coding=utf-8

import pandas as pd
import numpy as np
import os



def cohort_register(pk,df):
    regist=pd.DataFrame()
    data=df[pk]
    data['draw_value']=1
    data['appl_value']=data['first_tm_draw_appl_dt'].map(lambda x: 1 if not pd.isnull(x) else 0)
    total_reg=data.groupby(['limit_actv_tm']).size()
    total_reg.name='reg_total'
    regist['total']=total_reg
    reg_temp=pd.pivot_table(data=data,values='appl_value',\
               index='limit_actv_tm',columns='draw_appl_intrv',\
               aggfunc=np.sum)
    cohort_reg=reg_temp.reindex(columns=[i for i in range(0,31,1)]).fillna(0)
    cohort_30=regist.join(cohort_reg)
    return cohort_30


if __name__ == '__main__':
    path = 'C:\Users\Administrator\Desktop\liushi\compete_data\\finaly_data\\'
    file_save_path = 'C:\Users\Administrator\Desktop\liushi\\result\\'
    pk=['limit_actv_tm','first_tm_draw_appl_dt','draw_appl_intrv']
    dirs=os.listdir(path)
    result_cohort=pd.DataFrame()
    for fl in dirs:
        temp=pd.read_csv(path+fl)
        cohort_temp=cohort_register(df=temp,pk=pk)
        result_cohort=result_cohort.append(cohort_temp)
    os.chdir(file_save_path)
    result_cohort.to_excel('compete_draw30.xlsx',header=True)




pass