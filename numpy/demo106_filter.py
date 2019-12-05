from __future__ import unicode_literals
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

sample_rate, noised_sigs = wf.read(
    '../data/da_data/noised.wav')
print(sample_rate, noised_sigs.shape)
noised_sigs = noised_sigs / 2 ** 15

times = np.arange(len(noised_sigs)) / sample_rate
mp.figure('Filter', facecolor='lightgray')
mp.subplot(221)
mp.title('Time Domain', fontsize=16)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], noised_sigs[:178], c='orangered',
        label='Noised')
mp.legend()

# 基于傅里叶变换，获取音频频域信息，绘制音频频域的：频率/能量图像。
freqs = nf.fftfreq(times.size, 1 / sample_rate)
noised_ffts = nf.fft(noised_sigs)
noised_pows = np.abs(noised_ffts)
mp.subplot(222)
mp.title('Frequency Domain', fontsize=16)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.semilogy(freqs[freqs >= 0], noised_pows[
            freqs >= 0], c='limegreen', label='Noised')
mp.legend()

# 将低频噪声去除后绘制音频频域的：频率/能量图像。
fund_freq = freqs[noised_pows.argmax()]
noised_indices = np.where(freqs != fund_freq)
filter_ffts = noised_ffts.copy()
filter_ffts[noised_indices] = 0
filter_pows = np.abs(filter_ffts)

mp.subplot(224)
mp.xlabel('Frequency', fontsize=12)
mp.ylabel('Power', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(freqs[freqs >= 0], filter_pows[freqs >= 0],
        c='dodgerblue', label='Filter')
mp.legend()


filter_sigs = nf.ifft(filter_ffts).real
mp.subplot(223)
mp.xlabel('Time', fontsize=12)
mp.ylabel('Signal', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times[:178], filter_sigs[:178], c='hotpink', label='Filter')
mp.legend()

wf.write('filter.wav', sample_rate * 2,
         (filter_sigs * 2 ** 15).astype(np.int16))

mp.show()