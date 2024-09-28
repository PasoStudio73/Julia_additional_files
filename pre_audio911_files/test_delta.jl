using PyCall
using SoleAudio
using DSP

af = pyimport("audioflux")
librosa = pyimport("librosa")

sr_src = 16000
x, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100/WavFiles/common_voice_en_23616312.wav", sr=sr_src, mono=true)
FFTLength = 256
mel_num = 26

# convert to Float64
x = Float64.(x)

setup = signal_setup(
    sr=sr,
    # fft
    window_type=:hann,
    window_length=FFTLength,
    overlap_length=Int(round(FFTLength * 0.500)),
    window_norm=:false,
    # spectrum
    freq_range=Int[0, sr/2],
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
    log_energy_pos=:none,
    delta_window_length=9,
    delta_axe=2
)

data = signal_data(
    x=x
)

takeFFT(data, setup)

l_bft_obj = af.BFT(
    num=129,
    radix2_exp=Int64(log2(FFTLength)),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, FFTLength * 0.500),
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
l_spec_arr = l_bft_obj.bft(x, result_type=1)

data.stft.stft = l_spec_arr

mel_spectrogram(data, setup)
_mfcc(data, setup)