using PyCall
using SoleAudio

af = pyimport("audioflux")
librosa = pyimport("librosa")

sr_src = 16000
x, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100/WavFiles/common_voice_en_23616312.wav", sr=sr_src, mono=true)
FFTLength = 256
mel_num = 26

# setup and data structures definition
setup = signal_setup(
    sr=sr,

    # fft
    window_type=:hann,
    window_length=FFTLength,
    overlap_length=round(Integer, FFTLength * 0.500),
    window_norm=:false,

    # spectrum
    freq_range=[0, round(Integer, sr / 2)],
    spectrum_type=:power,

    # mel
    mel_style=:htk,
    num_bands=mel_num,
    filterbank_design_domain=:linear,
    filterbank_normalization=:bandwidth,
    frequency_scale=:mel,

    # mfcc
    mfcc_coeffs=13,
    rectification=:log,
    log_energy_pos=:replace,
    delta_window_length=9,

    # spectral
    spectral_spectrum=:linear
)

# convert to Float64
x = Float64.(x)
# preemphasis
# zi = 2 * x[1] - x[2]
# filt!(x, [1.0, -0.97], 1.0, x, [zi])
# normalize
# x = x ./ maximum(abs.(x))

data = signal_data(
    x=x
)
# zi = 2 * x[1] - x[2]
# filt!(x, [1.0, -0.97], 1.0, x, [zi])

takeFFT(data, setup)
# lin_spectrogram(data, setup)
mel_spectrogram(data, setup)
_mfcc(data, setup)
# spectral_features(data, setup)
# f0(data, setup)

# vcat((
#     data.mfcc_coeffs',
#     data.mel_spectrogram'
# )...)

data.mfcc_delta'
# data.mfcc_coeffs'