# setup
import librosa as librosa
import audioflux as af
import math as math
import numpy as np

sr = 8000

x, sr = librosa.load('/home/riccardopasini/.julia/dev/SoleAudio.jl/test/common_voice_en_23616312.wav', sr=sr)

FFTLength = 256
mel_num = 26

s_obj = af.BFT(
    num=mel_num,
    radix2_exp=int(math.log2(FFTLength)),
    samplate=sr,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HANN,
    slide_length=round(FFTLength * 0.500),
    scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
    data_type=af.type.SpectralDataType.MAG,
)
s_spec_arr = s_obj.bft(x)

s_spec_arr = np.abs(s_spec_arr)

s_spectral_obj = af.Spectral(
    num=s_obj.num,
    fre_band_arr=s_obj.get_fre_band_arr()
)
s_n_time = s_spec_arr.shape[-1]
s_spectral_obj.set_time_length(s_n_time)

centroid_arr = s_spectral_obj.centroid(s_spec_arr)