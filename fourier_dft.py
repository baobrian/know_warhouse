# coding=utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl


if __name__ == '__main__':

    path = 'E:\Data_temp\\20190902\data\\'
    dirs = os.listdir(path)
    simulation=np.ones(900,int)
    for fl in dirs:
        data=pd.read_csv(path+fl,usecols=['value'])
        c_data=np.array(data).flatten()
    t = np.arange(0, len(c_data),1)
    fft_size=50
    samples=len(c_data)
    simulation = np.ones(samples, int)
    frep=np.linspace(0,fft_size,fft_size+1)
    temp_data=c_data[:samples]
    # 开始画画
    pl.figure(figsize=(10, 6))
    pl.subplot(411)
    plt.grid(True)
    pl.plot(t[:], temp_data)
    # 开始画时域信号
    dft_data=np.abs(np.fft.rfft(temp_data)/fft_size)
    pl.subplot(412)
    pl.plot(frep[:fft_size/2+1],dft_data[:fft_size/2+1])
    # 开始画时域模拟信号
    pl.subplot(413)
    plt.grid(True)
    pl.plot(t[:], simulation)
    # 开始画模拟信号的频谱
    a_data=np.abs(np.fft.rfft(simulation)/fft_size)
    pl.subplot(414)
    pl.plot(frep[:fft_size/2+1],a_data[:fft_size/2+1])
    # 展现画布
    plt.show()







    # sampling_rate = 8000
    # fft_size = 512
    # t = np.arange(0, 1.0, 1.0 / sampling_rate)
    # x = np.sin(2 * np.pi * 156 * t) +2* np.sin(2 * np.pi * 234 * t)
    # b = x[:fft_size]

    # pl.figure(figsize=(10, 6))
    # pl.subplot(311)
    # #plt.plot(b)
    # plt.grid(True)
    # #plt.xlim(0, )
    # pl.plot(t[:fft_size], b)
    # freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)
    # dft_a = np.fft.rfft(b)/512
    # cc=np.abs(dft_a)
    # zz=np.clip(np.abs(dft_a), 1e-20, 1e100)
    #
    # pl.subplot(312)
    # pl.plot(freqs, cc)
    # #plt.plot(cc)
    #
    # pl.subplot(313)
    # pl.plot(freqs, zz)
    # # plt.plot(zz)
    # plt.grid(True)
    # plt.xlim(0, 100)
    # plt.show()
    pass

