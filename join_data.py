# coding=utf-8
import pandas as pd
import os
import columns_list as c_list


def join_data(path):
    result = pd.DataFrame()
    dirs = os.listdir(path)
    for file in dirs:
        temp=pd.read_csv(path+file)
        result=result.append(temp,ignore_index=True)
    return result

def map_func(para):

    # # if para>=0 and para<=3:
    # #     return 2
    # if para>=0 and para<=30:
    #     return 2
    # if para>=31 and para<=90:
    #     return 3
    # if para>90:
    #     return 4

    # if para <0:
    #     return -1
    if para>=0 and para<=28:
        return 1
    # if para>3:
    #     return 2
    else :
        return 0




def map_func2(para):


    if para==-1:
        return 1000
    else:
        return para




if __name__ == '__main__':
    path='C:\Users\Administrator\Desktop\liushi\data20190702\\vali\\'
    file_save_path='C:\Users\Administrator\Desktop\liushi\\data20190702\\'
    result=join_data(path=path)
    os.chdir(file_save_path)
    #c_result=result[c_list.re_draw_ll]
    result.to_csv('vali_all.csv',index=False)
    # result.to_csv('compete.csv',index=False)
    # result=result[(result['draw_intrv']>-1)]
    # print sum((result['draw_intrv']<0))
    # r_column = c_list.reg_column
    # result['regress']=result['draw_intrv']
    # r_result = result[r_column]
    # r_result.to_csv('result_reg.csv', index=False)

    pass

