using PyCall

af = pyimport("audioflux")
librosa = pyimport("librosa")
plt = pyimport("matplotlib.pyplot")

audio, sr = af.read("common_voice_en_14679.mp3", samplate=8000, is_mono=true)

## function elsif :matlab
# compute linear spectrogram
l_bft_obj = af.BFT(
    num=2049,
    radix2_exp=12,
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
# compute mel spectrogram
m_bft_obj = af.BFT(
    num=128,
    radix2_exp=12,
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.AREA,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
m_spec_arr = m_bft_obj.bft(audio, result_type=1)
# compute mel spectrogram for exporting
m_exp_bft_obj = af.BFT(
    num=128,
    radix2_exp=12,
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.AREA,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
m_exp_spec_arr = m_exp_bft_obj.bft(audio, result_type=1)
# compute bark spectrogram
b_bft_obj = af.BFT(
    num=128,
    radix2_exp=12,
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.BARK,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.NONE,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
b_spec_arr = b_bft_obj.bft(audio, result_type=1)
# compute erb spectrogram
e_bft_obj = af.BFT(
    num=128,
    radix2_exp=12,
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.ERB,
    style_type=af.type.SpectralFilterBankStyleType.GAMMATONE,
    normal_type=af.type.SpectralFilterBankNormalType.AREA,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
e_spec_arr = e_bft_obj.bft(audio, result_type=1)
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
# compute gtcc and deltasm_
g_xxcc_obj = af.XXCC(num=e_bft_obj.num)
g_xxcc_obj.set_time_length(time_length=length(e_spec_arr[2, :]))
g_spectral_obj = af.Spectral(
    num=e_bft_obj.num,
    fre_band_arr=e_bft_obj.get_fre_band_arr())
g_n_time = length(e_spec_arr[2, :])
g_spectral_obj.set_time_length(g_n_time)
g_energy_arr = g_spectral_obj.energy(g_spec_arr)
gtcc_arr, g_delta_arr, g_deltadelta_arr = g_xxcc_obj.xxcc_standard(
    e_spec_arr,
    g_energy_arr,
    cc_num=13,
    delta_window_length=9,
    energy_type=af.type.CepstralEnergyType.REPLACE,
    rectify_type=af.type.CepstralRectifyType.LOG
)
# compute spectral features
s_spec_arr = l_bft_obj.bft(audio)
s_phase_arr = af.utils.get_phase(s_spec_arr)
s_spec_arr = abs.(s_spec_arr)
spectral_obj = af.Spectral(
    num=l_bft_obj.num,
    fre_band_arr=l_bft_obj.get_fre_band_arr()
)
s_n_time = length(spec_arr[2, :])
spectral_obj.set_time_length(s_n_time)
# band_width_arr = spectral_obj.band_width(s_spec_arr)
# broadband_arr = spectral_obj.broadband(s_spec_arr)
# cd_arr = spectral_obj.cd(s_spec_arr, s_phase_arr)
centroid_arr = spectral_obj.centroid(s_spec_arr)
crest_arr = spectral_obj.crest(s_spec_arr)
decrease_arr = spectral_obj.decrease(s_spec_arr)
# eef_arr = spectral_obj.eef(s_spec_arr)
# eer_arr = spectral_obj.eer(s_spec_arr)
# energy_arr = spectral_obj.energy(s_spec_arr)
entropy_arr = spectral_obj.entropy(s_spec_arr)
flatness_arr = spectral_obj.flatness(s_spec_arr)
flux_arr = spectral_obj.flux(s_spec_arr)
# hfc_arr = spectral_obj.hfc(s_spec_arr)
kurtosis_arr = spectral_obj.kurtosis(s_spec_arr)
# max_arr, max_fre_arr = spectral_obj.max(s_spec_arr)
# mean_arr, mean_fre_arr = spectral_obj.mean(s_spec_arr)
# mkl_arr = spectral_obj.mkl(s_spec_arr)
# novelty_arr = spectral_obj.novelty(s_spec_arr)
# nwpd_arr = spectral_obj.nwpd(s_spec_arr, s_phase_arr)
# pd_arr = spectral_obj.pd(s_spec_arr, s_phase_arr)
# rcd_arr = spectral_obj.rcd(s_spec_arr, s_phase_arr)
# rms_arr = spectral_obj.rms(s_spec_arr)
rolloff_arr = spectral_obj.rolloff(s_spec_arr)
# sd_arr = spectral_obj.sd(s_spec_arr)
skewness_arr = spectral_obj.skewness(s_spec_arr)
slope_arr = spectral_obj.slope(s_spec_arr)
spread_arr = spectral_obj.spread(s_spec_arr)
# var_arr, var_fre_arr = spectral_obj.var(s_spec_arr)
# wpd_arr = spectral_obj.wpd(s_spec_arr, s_phase_arr)

## signal silence split
trim_i = librosa.effects.split(audio[:, end], top_db=2)
if (trim_i[1,1] == 0)
    trim_i[1,1] = 1
end
split_points = vec(trim_i')
signal_trim = reduce(vcat, [audio[split_points[i]+1:split_points[i+1]+1] for i in 1:2:length(split_points)])


if (split_points[1] == 1 && split_points[end] != length(audio))
    circshift!(split_points, -1)
    split_points[end] = length(audio)
elseif (split_points[1] != 1 && split_points[end] != length(audio))
    circshift!(split_points, 1)
    split_points = push!(Vector{Any}(split_points), split_points[1])
    split_points[1] = 1
    split_points = push!(split_points, length(audio))
elseif (split_points[1] != 1 && split_points[end] == length(audio))
    circshift!(split_points, 1)
    split_points[1] = 1
else
    silence_trim = 0
end


