#coding=utf-8

import  pandas as pd


a=pd.DataFrame()

df=pd.DataFrame({'value':[1,2,3,4],'name':list('abcd')})
cc=a.append(df)
print cc