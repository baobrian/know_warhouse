# *coding=utf-8


import cohort_anlysis as Coh
import data_info as Daf
import extract_data as exd
import os

def getdata():
    # 获取训练数据集
    sql=Daf.SQL
    print ('Step1 :  开始获取数据集......')
    downloader = exd.DownloadFromImpala()
    df = downloader.downloaddata(sql)
    print (' 获取数据集结束')
    return df

def data_analysis_main(para1,para2,para3,para4):
    data=getdata()
    cohort = Coh.CohortAnalysis(data=data, stat_time=para1, end_time=para2, boundary=para3)
    os.chdir(Daf.file_save_path)
    cohort.analysis_cohort(isPercent=para4)
    return


if __name__ == '__main__':
    data_analysis_main(Daf.para1,Daf.para2,Daf.para3,Daf.para4)
    print '同期群分析文件已生成'




