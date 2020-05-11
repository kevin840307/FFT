import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal

# sin
#x = np.arange(0, 1.0, 1.0 / 128)
#y = np.sin(2 * np.pi * 1 * x)
#plt.plot(x, y, 'r')

#Amplitude
#x = np.arange(0, 1.0, 1.0 / 128)
#y = np.sin(2 * np.pi * 1 * x)
#plt.plot(x, y, 'r')

#x = np.arange(0, 1.0, 1.0 / 128)
#y = 2 * np.sin(2 * np.pi * 1 * x)
#plt.plot(x, y, 'b')

# Phase
#x = np.arange(0.2, 1.0, 1.0 / 128)
#y = np.sin(2 * np.pi * 1 * x)
#plt.plot(x, y, 'r')

#x = np.arange(0.2, 1.0, 1.0 / 128)
#y = 2 * np.sin(2 * np.pi * 1 * x)
#plt.plot(x, y, 'b')

# Frequency 
#x = np.arange(0, 1.0, 1.0 / 8000)
#y = np.sin(2 * np.pi * 1 * x)
#plt.plot(x, y, 'r')

#x = np.arange(0, 1.0, 1.0 / 8000)
#y = np.sin(2 * np.pi * 4000 * x)
#plt.plot(x, y, 'b')

#plt.show()




# 疊加
#def add_sin(x, n):
#    y = np.sin(x)
#    for n in range(3, n + 1, 2):
#        y += 4 * np.sin(n * x) / (n * np.pi)
#    return y

#plt.ion()
#plt.show()

#N = 5
#angle = np.arange(0, 2 * np.pi + 0.05, 0.05)
#last_x = angle[0]
#last_y = add_sin(last_x, N)
#ln_list = [None for _ in range(0, N * 2 + 1, 2)]
#for index in range(1, len(angle), 1):
#    x = angle[index]
#    y = add_sin(x, N)

#    plt.subplot(122)
#    plt.plot([last_x, x], [last_y, y], 'r')
#    last_x = x
#    last_y = y


#    plt.subplot(121)
#    for ln in ln_list:
#        if ln != None:
#            ln.remove()
#    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#    circle_offset_x = 0
#    circle_offset_y = 0

#    for n in range(1, N + 1, 2):
#        color = colors[n // 2]

#        pos = np.arange(-np.pi, np.pi + 1, 0.05)
#        circle_x = circle_offset_x + 4 * np.cos(n * pos) / (n * np.pi)
#        circle_y = circle_offset_y + 4 * np.sin(n * pos) / (n * np.pi)
#        ln_list[n - 1], = plt.plot(circle_x, circle_y, color)

#        end_x = circle_offset_x + 4 * np.cos(n * angle[index]) / (n * np.pi)
#        end_y = circle_offset_y + 4 * np.sin(n * angle[index]) / (n * np.pi)
#        ln_list[n], = plt.plot([circle_offset_x, end_x],[circle_offset_y, end_y], color)

#        circle_offset_x = end_x
#        circle_offset_y = end_y

#        if n == N:
#            plt.scatter([end_x], [end_y], c=color)

#    plt.draw()
#    plt.pause(0.001)
#plt.pause(9999)





# e^ipi
#plt.ion()
#plt.show()
#plt.subplot(121)
#pos = np.arange(-np.pi, np.pi + 1, 0.05)
#y = np.cos(pos)
#x = np.sin(pos)
#plt.plot(x, y, 'r')

#ln = None
#N = 100
#offset = 2 * np.pi / N
#Hz = 1
#angle = np.arange(-np.pi, np.pi, offset)
#for i in range(len(angle)):
#    plt.subplot(121)
#    if ln != None:
#        ln.remove()
#    e = angle[i]
#    y = np.cos(Hz * e)
#    x = np.sin(Hz * e)
#    ln, = plt.plot([0,x],[0,y], 'b--')

#    plt.subplot(122)
#    last_x = e -  offset / 2
#    last_y = np.sin(Hz * last_x)
#    x = e
#    y = np.sin(Hz * x)
#    plt.plot([last_x, x], [last_y, y], 'r')

#    last_x = e - offset / 2
#    last_y = np.cos(Hz * last_x)
#    x = e
#    y = np.cos(Hz * x)
#    plt.plot([last_x, x], [last_y, y], 'b')

#    plt.draw()
#    plt.pause(0.01)
#plt.pause(999)




# FFT
#x = np.arange(-10, 10, 0.1)
#value = np.sin(x) + np.sin(3 * x) + np.sin(5 * x)
#matrix = value
#matrix_dct = np.fft.rfft(matrix)
#matrix_dct = np.fft.fftshift(matrix_dct)

#plt.subplot(221)
#plt.plot(x, value, 'r')

#plt.subplot(222)
#plt.plot(np.arange(0, len(matrix_dct)), np.log10(matrix_dct) * 20, 'r')

#matrix_dct[50: 56] = 0

#plt.subplot(223)
#print(len(np.fft.irfft(np.fft.fftshift(matrix_dct))))
#plt.plot(x, np.fft.irfft(np.fft.fftshift(matrix_dct)), 'r')

#plt.subplot(224)
#plt.plot(np.arange(0, len(matrix_dct)), np.log10(matrix_dct) * 20, 'r')

#plt.show()

#import pylab as pl
#sampling_rate = 8000
#fft_size = 512
#t = np.arange(0, 1.0, 1.0/sampling_rate)
#x = np.sin(2*np.pi*300*t)  + 2 * np.sin(2*np.pi*500*t) + np.cos(2 * np.pi*100*t) + 5 * np.cos(2 * np.pi * 50 *t)
#xs = x[:fft_size]
#xf = np.fft.rfft(xs)/fft_size
#freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
#xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
#pl.figure(figsize=(8,4))
#pl.subplot(211)
#pl.plot(t[:fft_size], xs)
#pl.xlabel("time(s)")
#pl.subplot(212)
#pl.plot(freqs, xfp)
#pl.xlabel(u"Hz")
#pl.subplots_adjust(hspace=0.4)
#pl.show()





# 3D
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.set_xlim([-10, 10])
#ax.set_ylim([0, 10])
#ax.set_zlim([-2, 2])
#N = 5
#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#for n in range(1, N + 1, 2):
#    color = colors[n // 2]
#    x = np.arange(-10, 10, 0.1)
#    y = np.ones_like(x) * n + 1
#    z = np.sin(n * x)
#    ax.plot(x, y, z, color=color)

#plt.show()