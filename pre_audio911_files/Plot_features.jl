using PyCall

af = pyimport("audioflux")
librosa = pyimport("librosa")
plt = pyimport("matplotlib.pyplot")

audio, sr = af.read("test.wav", samplate=8000, is_mono=true)

fft = 256
mel_num = 26

m_bft_exp_obj = af.BFT(
    num=mel_num,
    radix2_exp=Int64(log2(fft)),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, fft * 0.500),
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    style_type=af.type.SpectralFilterBankStyleType.GAUSS,
    normal_type=af.type.SpectralFilterBankNormalType.BAND_WIDTH,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
m_exp_spec_arr = m_bft_exp_obj.bft(audio, result_type=1)

librosa.display.specshow(m_exp_spec_arr)
plt.savefig("mel_spec.jpg")
plt.figure().clear()