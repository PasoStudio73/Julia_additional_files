using PyCall

af = pyimport("audioflux")
librosa = pyimport("librosa")

sr_src = 16000
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100/WavFiles/common_voice_en_23616312.wav", sr=sr_src, mono=true)
FFTLength = 256
mel_num = 26

# convert to Float64
audio = Float64.(audio)

# compute mel spectrogram
m_bft_obj = af.BFT(
    num=mel_num,
    radix2_exp=Int64(log2(FFTLength)),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2, # settato a 4000
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, FFTLength * 0.500),
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.BAND_WIDTH,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
m_spec_arr = m_bft_obj.bft(audio, result_type=1)

m_bft_exp_obj = af.BFT(
    num=mel_num,
    radix2_exp=Int64(log2(FFTLength)),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, FFTLength * 0.500),
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.BAND_WIDTH,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
m_exp_spec_arr = m_bft_exp_obj.bft(audio, result_type=1)

# compute mfcc and deltas
m_xxcc_obj = af.XXCC(num=m_bft_obj.num)
m_xxcc_obj.set_time_length(time_length=length(m_spec_arr[2, :]))
m_spectral_obj = af.Spectral(
    num=m_bft_obj.num,
    fre_band_arr=m_bft_obj.get_fre_band_arr())
m_n_time = length(m_spec_arr[2, :])
m_spectral_obj.set_time_length(m_n_time)
m_energy_arr = m_spectral_obj.energy(m_spec_arr)
mfcc_arr, m_delta_arr, m_deltadelta_arr = m_xxcc_obj.xxcc_standard(
    m_spec_arr,
    m_energy_arr,
    cc_num=13,
    delta_window_length=9,
    energy_type=af.type.CepstralEnergyType.REPLACE,
    rectify_type=af.type.CepstralRectifyType.LOG
)

# # compute spectral features
# s_obj = af.BFT(
#     num=mel_num,
#     radix2_exp=Int64(log2(FFTLength)),
#     samplate=sr,
#     high_fre=sr / 2,
#     window_type=af.type.WindowType.HANN,
#     slide_length=round(Integer, FFTLength * 0.500),
#     scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
#     data_type=af.type.SpectralDataType.MAG,
# )
# s_spec_arr = s_obj.bft(audio)
# s_spec_arr = abs.(s_spec_arr)

# s_spectral_obj = af.Spectral(
#     num=s_obj.num,
#     fre_band_arr=s_obj.get_fre_band_arr()
# )
# s_n_time = length(s_spec_arr[2, :])
# s_spectral_obj.set_time_length(s_n_time)

# centroid_arr = s_spectral_obj.centroid(s_spec_arr)
# crest_arr = s_spectral_obj.crest(s_spec_arr)
# decrease_arr = s_spectral_obj.decrease(s_spec_arr)
# flatness_arr = s_spectral_obj.flatness(s_spec_arr)
# flux_arr = s_spectral_obj.flux(s_spec_arr)
# kurtosis_arr = s_spectral_obj.kurtosis(s_spec_arr)
# rolloff_arr = s_spectral_obj.rolloff(s_spec_arr)
# skewness_arr = s_spectral_obj.skewness(s_spec_arr)
# slope_arr = s_spectral_obj.slope(s_spec_arr)
# spread_arr = s_spectral_obj.spread(s_spec_arr)

# vcat((
#     mfcc_arr,
#     # m_delta_arr,
#     # m_deltadelta_arr,
#     # centroid_arr',
#     # crest_arr',
#     # decrease_arr',
#     # # entropy_arr',
#     # flatness_arr',
#     # flux_arr',
#     # kurtosis_arr',
#     # rolloff_arr',
#     # skewness_arr',
#     # slope_arr',
#     # spread_arr',
#     m_exp_spec_arr
# )...)

# m_delta_arr
mfcc_arr