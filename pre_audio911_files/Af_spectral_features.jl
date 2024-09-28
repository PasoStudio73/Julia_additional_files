using PyCall

af = pyimport("audioflux")
librosa = pyimport("librosa")
plt = pyimport("matplotlib.pyplot")

audio, sr = af.read("test.wav", samplate=8000, is_mono=true)

WIN_TIME = 0.025 ## Window length in sec. Default is 0.025
STEP_TIME = 0.01  ## Step between successive windows in sec. Default is 0.01

fft = 1024
mel_num = 128

# compute spectral features
s_obj = af.BFT(
    num=mel_num,
    radix2_exp=Int64(log2(fft)),
    samplate=sr,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    data_type=af.type.SpectralDataType.MAG,
)
s_spec_arr = s_obj.bft(audio)
s_phase_arr = af.utils.get_phase(s_spec_arr)
s_spec_arr = abs.(s_spec_arr)

s_spectral_obj = af.Spectral(
    num=s_obj.num,
    fre_band_arr=s_obj.get_fre_band_arr()
)
s_n_time = length(s_spec_arr[2, :])
s_spectral_obj.set_time_length(s_n_time)


# s_spec_arr = abs.(s_spec_arr)

centroid_arr = s_spectral_obj.centroid(s_spec_arr)
crest_arr = s_spectral_obj.crest(s_spec_arr)
decrease_arr = s_spectral_obj.decrease(s_spec_arr)
entropy_arr = s_spectral_obj.entropy(s_spec_arr)
flatness_arr = s_spectral_obj.flatness(s_spec_arr)
flux_arr = s_spectral_obj.flux(s_spec_arr)
kurtosis_arr = s_spectral_obj.kurtosis(s_spec_arr)
rolloff_arr = s_spectral_obj.rolloff(s_spec_arr)
skewness_arr = s_spectral_obj.skewness(s_spec_arr)
slope_arr = s_spectral_obj.slope(s_spec_arr)
spread_arr = s_spectral_obj.spread(s_spec_arr)


# librosa.display.specshow(s_spec_arr)
# plt.show()

librosa.display.waveshow(spread_arr')
plt.show()