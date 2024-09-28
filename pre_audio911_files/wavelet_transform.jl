using PyCall
using StaticArrays: SVector

librosa = pyimport("librosa")
soundfile = pyimport("soundfile")
wt = pyimport("pywt")
plt = pyimport("matplotlib.pyplot")

fft = 1024
mel_num = 26
WIN_TIME = 0.025
STEP_TIME = 0.01
sr_src = 8000
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav", sr=sr_src, mono=true)

w = wt.Wavelet("db2")
max_level = wt.dwt_max_level(data_len=length(audio), filter_len=w.dec_len)
## Chack audio length
level = max_level
pad = length(audio) % (2^level)
audio_pad = audio[1:length(audio)-pad]
## compute wavelets
coeffs = wt.swt(
    data=audio_pad, wavelet="db2",
    level=max_level,
    trim_approx=true,
)
swt_spec = reshape(collect(Iterators.flatten(coeffs)), (lastindex(coeffs[1]), length(coeffs)))

plt.figure().clear()
librosa.display.specshow(swt_spec')
plt.show()

l_bft_obj = af.BFT(
    num=mel_num,
    radix2_exp=Int64(log2(fft)),
    samplate=sr,
    low_fre=0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
l_spec_arr = l_bft_obj.bft(audio, result_type=1)

plt.figure().clear()
librosa.display.specshow(l_spec_arr)
plt.show()

cwt_obj = af.CWT(
    num=mel_num,
    radix2_exp=Int64(log2(fft)),
    samplate=sr,
    bin_per_octave=12,
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    wavelet_type=af.type.WaveletContinueType.MORLET
)
cwt_spec = abs.(cwt_obj.cwt(audio))

plt.figure().clear()
librosa.display.specshow(cwt_spec)
plt.show()


cwt_spec, freqs = wt.cwt(
    data=audio,
    scales=range(1, 128),
    wavelet="morl",
    sampling_period=1 / sr
)

plt.figure().clear()
librosa.display.specshow(cwt_spec')
plt.show()

wpt = wt.WaveletPacket(
    data=audio,
    wavelet = "db2"
)

cA, cD = wt.dwt(
    data=audio,
    wavelet="db1"
)

plt.figure().clear()
librosa.display.waveshow(cD)
plt.show()