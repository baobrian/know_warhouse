import pandas as pd


data={'m1':[1,23,4],'m2':[1,1,1],'m3':[3,3,3]}
df=pd.DataFrame(data)
gg=df.reindex(columns=['m1','s1','m2'])
print df
print gg