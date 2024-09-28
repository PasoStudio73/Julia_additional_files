# common_voice_en_14679

using PyCall
using MFCC
using DSP
using Arrow
using DataFrames

af = pyimport("audioflux")

librosa = pyimport("librosa")
plt = pyimport("matplotlib.pyplot")

# WIN_TIME = 0.025 ## Window length in sec.
WIN_TIME = 0.032
STEP_TIME = 0.01  ## Step between successive windows in sec.
NUM_CEPS = 13   ## Number of cepstra to return. Default is 13
PRE_EMPH = 0.97 ## Apply pre-emphasis filter. Default is 0.97

audio, sr = af.read("/Users/riccardopasini/results/speech/Search New Audio Features/spec_test.wav", samplate=8000, is_mono=true)
# af.write("af.wav", audio, samplate=sr)
nwin = round(Integer, WIN_TIME * sr)
nstep = round(Integer, STEP_TIME * sr)
nfft = 2 .^ Integer((ceil(log2(nwin))))
window = hamming(nwin)
noverlap = nwin - nstep

################# Spectral related ######################


# pspec = spectrogram(
#     audio .* (1<<15), 
#     nwin, 
#     noverlap, 
#     nfft=nfft, 
#     fs=sr, 
#     window=window, 
#     onesided=true
# ).power

## Short-time Fourier transform (Linear/STFT)
lspec, frq_lspec = af.linear_spectrogram(
    audio,
    samplate=sr,
    radix2_exp=Integer((ceil(log2(nwin)))),
    low_fre=0.0,
    slide_length=nstep,
    window_type=af.type.WindowType.HAMM,
    data_type=af.type.SpectralDataType.POWER #MAG
)
lspec = af.utils.power_to_db(lspec)

librosa.display.specshow(lspec)
plt.show()

## Mel-scale spectrogram.
mspec, frq_mspec = af.mel_spectrogram(
    audio,
    samplate=sr,
    num=20,
    radix2_exp=Integer((ceil(log2(nwin)))),
    # low_fre=0.0,
    # high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=nstep
    # style_type=af.type.SpectralFilterBankStyleType.SLANEY, ETSI, GAMMATONE, POINT, RECT, HANN, HAMM, BLACKMAN, BOHMAN, KAISER, GAUSS
    # normal_type=af.type.SpectralFilterBankNormalType.NONE, AREA, BAND_WIDTH
    # data_type=af.type.SpectralDataType.POWER, MAG
)
mspec = af.utils.power_to_db(mspec)

librosa.display.specshow(mspec)
plt.show()

## Bark-scale spectrogram.
bspec, frq_bspec = af.bark_spectrogram(
    audio,
    samplate=sr,
    num=24,
    radix2_exp=Integer((ceil(log2(nwin)))),
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=nstep
    # style_type=af.type.SpectralFilterBankStyleType.SLANEY, ETSI, GAMMATONE, POINT, RECT, HANN, HAMM, BLACKMAN, BOHMAN, KAISER, GAUSS
    # normal_type=af.type.SpectralFilterBankNormalType.NONE, AREA, BAND_WIDTH
    # data_type=af.type.SpectralDataType.POWER, MAG
)
bspec = af.utils.power_to_db(bspec)

librosa.display.specshow(bspec)
plt.show()

## Erb-scale spectrogram.
espec, frq_espec = af.erb_spectrogram(
    audio,
    samplate=sr,
    num=24,
    radix2_exp=Integer((ceil(log2(nwin)))),
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=nstep
    # style_type=af.type.SpectralFilterBankStyleType.SLANEY, ETSI, GAMMATONE, POINT, RECT, HANN, HAMM, BLACKMAN, BOHMAN, KAISER, GAUSS
    # normal_type=af.type.SpectralFilterBankNormalType.NONE, AREA, BAND_WIDTH
    # data_type=af.type.SpectralDataType.POWER, MAG
)
espec = af.utils.power_to_db(espec)

librosa.display.specshow(espec)
plt.show()

## Constant-Q transform (CQT)
cqtspec, freq_cqtspec = af.cqt(
    audio,
    num=84, #default, 7*12
    samplate=sr,
    bin_per_octave=12,
    # factor=float,
    # beta=float,
    # thresh=float,
    window_type=af.type.WindowType.HANN,
    slide_length=round(Integer, STEP_TIME * sr)
)
cqtspec = af.utils.power_to_db(cqtspec)

librosa.display.specshow(cqtspec)
plt.show()

## Variable-Q transform (VQT)
vqtspec, freq_vqtspec = af.vqt(
    audio,
    num=84, #default, 7*12
    samplate=sr,
    bin_per_octave=12,
    # factor=float,
    # beta=float,
    # thresh=float,
    window_type=af.type.WindowType.HAMM,
    slide_length=nstep,
    is_scale=false
)
vqtspec = af.utils.power_to_db(vqtspec)

librosa.display.specshow(vqtspec)
plt.show()

#################### Cepstral coefficients ########################

mfcc_parameters = (;
    wintime=WIN_TIME,
    steptime=STEP_TIME,
    numcep=NUM_CEPS,
    lifterexp=-22,
    sumpower=false,
    preemph=0.94,
    dither=false,
    minfreq=0.0,
    maxfreq=sr / 2,
    nbands=20,
    bwidth=1.0,
    dcttype=3,
    fbtype=:htkmel,
    usecmp=false,
    modelorder=0
)
jl_mfccs = mfcc(audio, sr; mfcc_parameters...)[1]'

## Mel-frequency cepstral coefficients (MFCCs)
mfccs, freq_mfccs = af.mfcc(
    audio,
    samplate=sr,
    cc_num=13,
    mel_num=26,
    rectify_type=af.type.CepstralRectifyType.LOG,
    radix2_exp=12,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=round(Integer, STEP_TIME * sr)
)

librosa.display.specshow(mfccs)
plt.show()

##Bark-frequency cepstral coefficients (BFCCs)
bfccs, freq_bfccs = af.bfcc(
    audio,
    samplate=sr,
    cc_num=NUM_CEPS,
    bark_num=26,
    rectify_type=af.type.CepstralRectifyType.LOG,
    radix2_exp=12,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=round(Integer, STEP_TIME * sr)
)

librosa.display.specshow(bfccs)
plt.show()

## Gammatone cepstral coefficients (GTCCs)
gtccs, freq_gtccs = af.gtcc(
    audio,
    samplate=sr,
    cc_num=13,
    erb_num=128, # default: 128
    radix2_exp=12,
    low_fre=0.0,
    high_fre=sr / 2,
    rectify_type=af.type.CepstralRectifyType.LOG,
    window_type=af.type.WindowType.HAMM, #default: HANN
    slide_length=round(Integer, STEP_TIME * sr),
)

librosa.display.specshow(gtccs)
plt.show()

## Constant-Q cepstral coefficients (CQCCs)
cqccs, freq_cqccs = af.cqcc(
    audio,
    samplate=sr,
    cc_num=13,
    cqt_num=84, # default: 84
    bin_per_octave=12,
    rectify_type=af.type.CepstralRectifyType.LOG,
    low_fre=80.0,
    window_type=af.type.WindowType.HAMM, #default: HANN
    slide_length=round(Integer, STEP_TIME * sr),
    normal_type=af.type.SpectralFilterBankNormalType.AREA,
    is_scale=false
)

cqccs, freq_cqccs = af.cqcc(
    audio,
    samplate=sr,
    cc_num=13,
    cqt_num=84, # default: 84
    bin_per_octave=12,
    rectify_type=af.type.CepstralRectifyType.LOG,
    low_fre=40.0,
    window_type=af.type.WindowType.HAMM, #default: HANN
    slide_length=round(Integer, STEP_TIME * sr),
    normal_type=af.type.SpectralFilterBankNormalType.AREA,
    is_scale=false
)

librosa.display.specshow(cqccs)
plt.show()

################# Chroma ###################

## Linear(STFT) chromagram
chroma_l = af.chroma_linear(
    audio,
    samplate=sr,
    # chroma_num=13,
    radix2_exp=12,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=round(Integer, STEP_TIME * sr),
    style_type=af.type.SpectralFilterBankStyleType.SLANEY, # ETSI, GAMMATONE, POINT, RECT, HANN, HAMM, BLACKMAN, BOHMAN, KAISER, GAUSS
    normal_type=af.type.SpectralFilterBankNormalType.NONE, #, AREA, BAND_WIDTH
    data_type=af.type.SpectralDataType.POWER #, MAG
)

librosa.display.specshow(chroma_l)
plt.show()

## Octave chromagram
chroma_o = af.chroma_octave(
    audio,
    samplate=sr,
    # chroma_num=13,
    # radix2_exp=Integer((ceil(log2(nwin)))),
    window_type=af.type.WindowType.HAMM,
    slide_length=nstep
    # style_type=af.type.SpectralFilterBankStyleType.SLANEY, ETSI, GAMMATONE, POINT, RECT, HANN, HAMM, BLACKMAN, BOHMAN, KAISER, GAUSS
    # normal_type=af.type.SpectralFilterBankNormalType.NONE, AREA, BAND_WIDTH
    # data_type=af.type.SpectralDataType.POWER, MAG
)

librosa.display.specshow(chroma_o)
plt.show()

## Constant-Q chromagram
chroma_c = af.chroma_cqt(
    audio,
    samplate=sr,
    # chroma_num=13,
    # num=84,
    # bin_per_octave=12,
    # # factor=float,
    # # beta=float,
    # # thresh=float,
    # window_type=af.type.WindowType.HAMM,
    # slide_length=nstep,
    # is_scale=false
)

librosa.display.specshow(chroma_c)
plt.show()

"""
## rasta.jl, function powspec
## window, step and fft sizes @96000
nwin = round(Integer, 0.025 * 96000) = 2400
nstep = round(Integer, 0.01 * 96000) = 960
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 4096
## window, step and fft sizes @88200
nwin = round(Integer, 0.025 * 88200) = 2205
nstep = round(Integer, 0.01 * 88200) = 882
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 4096
## window, step and fft sizes @48000
nwin = round(Integer, 0.025 * 48000) = 1200
nstep = round(Integer, 0.01 * 48000) = 480
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 2048
## window, step and fft sizes @44100
nwin = round(Integer, 0.025 * 44100) = 1102
nstep = round(Integer, 0.01 * 44100) = 441
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 2048
## window, step and fft sizes @32000
nwin = round(Integer, 0.025 * 32000) = 800
nstep = round(Integer, 0.01 * 32000) = 320
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 1024
## window, step and fft sizes @16000
nwin = round(Integer, 0.025 * 16000) = 400
nstep = round(Integer, 0.01 * 16000) = 160
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 512
## window, step and fft sizes @8000
nwin = round(Integer, 0.025 * 8000) = 200
nstep = round(Integer, 0.01 * 8000) = 80
nfft = 2 .^ Integer((ceil(log2(nwin)))) = 256
"""

############### Transforms ##################

bft_obj = af.BFT(
    num=17,
    radix2_exp=Integer((ceil(log2(nwin)))),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=nstep,
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    data_type=af.type.SpectralDataType.POWER
)
bftspec = af.utils.power_to_db(abs.(bft_obj.bft(audio, result_type=1)))

librosa.display.specshow(bftspec)
plt.show()

nsgt_obj = af.NSGT(
    num=84,
    radix2_exp=Integer((ceil(log2(nwin)))),
    samplate=sr,
    bin_per_octave=12,
    scale_type=af.type.SpectralFilterBankScaleType.OCTAVE,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.NONE
)
nsgtspec = abs.(nsgt_obj.nsgt(audio))

librosa.display.specshow(nsgtspec)
plt.show()

cwt_obj = af.CWT(
    num=48,
    radix2_exp=8,
    samplate=sr,
    bin_per_octave=12,
    wavelet_type=af.type.WaveletContinueType.MORSE,
    scale_type=af.type.SpectralFilterBankScaleType.OCTAVE
)
cwtspec = abs.(cwt_obj.cwt(audio))

librosa.display.specshow(cwtspec)
plt.show()

pwt_obj = af.PWT(
    num=84,
    radix2_exp=Integer((ceil(log2(nwin)))),
    samplate=sr,
    bin_per_octave=12,
    scale_type=af.type.SpectralFilterBankScaleType.OCTAVE,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.NONE
)
pwtspec = abs.(pwt_obj.pwt(audio))

librosa.display.specshow(pwtspec)
plt.show()

#################### XXCC #######################
bft_obj = af.BFT(
    num=17,
    radix2_exp=Integer((ceil(log2(nwin)))),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=nstep,
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    data_type=af.type.SpectralDataType.POWER
)
bftspec = abs.(bft_obj.bft(audio))

xxcc_obj = af.XXCC(bft_obj.num)
xxcc_obj.set_time_length(time_length=length(bftspec[2, :]))
mfcc_arr = xxcc_obj.xxcc(bftspec)

librosa.display.specshow(mfcc_arr)
plt.show()

fe_obj = af.FeatureExtractor()

########################## spectral features ###############################

bft_obj = af.BFT(
    num=26,
    radix2_exp=12,
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    data_type=af.type.SpectralDataType.POWER
)
spec_arr = bft_obj.bft(audio)
phase_arr = af.utils.get_phase(spec_arr)
spec_arr = abs.(spec_arr)

spectral_obj = af.Spectral(
    num=bft_obj.num,
    fre_band_arr=bft_obj.get_fre_band_arr()
)

n_time = length(spec_arr[2, :])
spectral_obj.set_time_length(n_time)

band_width_arr = spectral_obj.band_width(spec_arr)
broadband_arr = spectral_obj.broadband(spec_arr)
cd_arr = spectral_obj.cd(spec_arr, phase_arr)
centroid_arr = spectral_obj.centroid(spec_arr)
crest_arr = spectral_obj.crest(spec_arr)
decrease_arr = spectral_obj.decrease(spec_arr)
eef_arr = spectral_obj.eef(spec_arr)
eer_arr = spectral_obj.eer(spec_arr)
energy_arr = spectral_obj.energy(spec_arr)
entropy_arr = spectral_obj.entropy(spec_arr)
flatness_arr = spectral_obj.flatness(spec_arr)
flux_arr = spectral_obj.flux(spec_arr)
hfc_arr = spectral_obj.hfc(spec_arr)
kurtosis_arr = spectral_obj.kurtosis(spec_arr)
max_arr = spectral_obj.max(spec_arr)
mean_arr = spectral_obj.mean(spec_arr)
mkl_arr = spectral_obj.mkl(spec_arr)
novelty_arr = spectral_obj.novelty(spec_arr)
nwpd_arr = spectral_obj.nwpd(spec_arr, phase_arr)
pd_arr = spectral_obj.pd(spec_arr, phase_arr)
rcd_arr = spectral_obj.rcd(spec_arr, phase_arr)
rms_arr = spectral_obj.rms(spec_arr)
rolloff_arr = spectral_obj.rolloff(spec_arr)
sd_arr = spectral_obj.sd(spec_arr)
skewness_arr = spectral_obj.skewness(spec_arr)
slope_arr = spectral_obj.slope(spec_arr)
spread_arr = spectral_obj.spread(spec_arr)
var_arr = spectral_obj.var(spec_arr)
wpd_arr = spectral_obj.wpd(spec_arr, phase_arr)




