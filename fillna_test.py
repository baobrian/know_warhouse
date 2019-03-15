#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np



data={'is_over':['1','2','3',np.nan,np.nan],
      'this_over_days':[np.nan,'2','3','3',np.nan]}
com='collection_result'
df=pd.DataFrame(data=data)
df1=df.fillna('0')
df2=df.fillna({'is_over':'2','this_over_days':'0'})
print '文件 %s 的字段分别为%s'%(com,df2.columns)
pass