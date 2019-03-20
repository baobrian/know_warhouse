#! /usr/bin/env python
# coding=utf-8

import  pandas as pd
file1=r'C:\Users\Administrator\Desktop\validation\validation\20190320\Collection_classify_result_20190312validate20190320.csv'
file2=r'C:\Users\Administrator\Desktop\validation\validation\20190320\Collection_regression_result_20190312validate20190320.csv'
print 'classify' in file1
print 'classify' in file2
data1=pd.read_csv(file1,names=['id_no','ps_due_dt','loan_no','bad_prob','rank','is_over','id_o_6'])
data2=pd.read_csv(file2,names=['id_no','ps_due_dt','laon_no','bad_prob','rank','is_'])

pass