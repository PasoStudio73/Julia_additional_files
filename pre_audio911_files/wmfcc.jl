using PyCall

librosa = pyimport("librosa")
af = pyimport("audioflux")
plt = pyimport("matplotlib.pyplot")

fft = 256
mel_num = 26
sr_src = 8000
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav", sr=sr_src, mono=true)

## DWT
obj = af.DWT(
    num=11, 
    radix2_exp=12, 
    samplate=sr,
    wavelet_type=af.type.WaveletDiscreteType.DB,
    t1=4, 
    t2=0
)
coef_arr, m_data_arr = obj.dwt(audio)
m_data_arr = abs.(m_data_arr)

## pyplot
librosa.display.specshow(m_data_arr)
plt.show()

## WPT
obj = af.WPT(
    num=7, 
    radix2_exp=12, 
    samplate=sr,
    wavelet_type=af.type.WaveletDiscreteType.SYM,
    t1=4,
    t2=0
)
coef_arr, m_data_arr = obj.wpt(audio)
m_data_arr = abs.(m_data_arr)

## pyplot
librosa.display.specshow(m_data_arr)
plt.show()