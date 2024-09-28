using PyCall

librosa = pyimport("librosa")
libf0 = pyimport("libf0.swipe")
af = pyimport("audioflux")

sr_src = 8000
audio, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav", sr=sr_src, mono=true)
STEP_TIME = 0.01  ## Step between successive windows in sec. Default is 0.01
fft = 1024

m_exp_bft_obj = af.BFT(
    num=26,
    radix2_exp=Int64(log2(fft)),
    samplate=sr,
    low_fre=0.0,
    high_fre=sr / 2,
    window_type=af.type.WindowType.HAMM,
    slide_length=round(Integer, STEP_TIME * sr),
    scale_type=af.type.SpectralFilterBankScaleType.MEL,
    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
    normal_type=af.type.SpectralFilterBankNormalType.AREA,
    data_type=af.type.SpectralDataType.POWER,
    is_reassign=false
)
m_exp_spec_arr = m_exp_bft_obj.bft(audio, result_type=1)

f0, t, strength = libf0.swipe(
    x=audio,
    Fs=sr,
    H=round(Integer, STEP_TIME * sr),
    F_min=55.0,
    F_max=1760.0,
)
f0 = f0[1:size(m_exp_spec_arr, 2)]
t = t[1:size(m_exp_spec_arr, 2)]
strength = strength[1:size(m_exp_spec_arr, 2)]