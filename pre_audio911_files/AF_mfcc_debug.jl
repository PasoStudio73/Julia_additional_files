using PyCall
using SoleAudio
using MFCC

af = pyimport("audioflux")
librosa = pyimport("librosa")
scipy = pyimport("scipy")
np = pyimport("numpy")

sr_src = 16000
x, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100/WavFiles/common_voice_en_23616312.wav", sr=sr_src, mono=true)
FFTLength = 256
mel_num = 26

# convert to Float64
x = Float64.(x)

setup = signal_setup(
    sr=sr,
    # fft
    window_type=[:hann, :periodic],
    window_length=FFTLength,
    overlap_length=Int(round(FFTLength * 0.500)),
    window_norm=:false,
    # spectrum
    freq_range=Int[0, sr/2],
    spectrum_type=:power,
    # mel
    mel_style=:htk,
    mel_bands=mel_num,
    filterbank_design_domain=:linear,
    filterbank_normalization=:bandwidth,
    frequency_scale=:mel,
    # mfcc
    mfcc_coeffs=13,
    rectification=:log,
    log_energy_pos=:none,
    delta_window_length=9
)

data = signal_data(
    x=x
)

takeFFT(data, setup)
mel_spectrogram(data, setup)

function create_DCT_matrix(
    mfcc_coeffs::Int64,
    time_length::Int64,
)
    # create DCT matrix
    matrix = zeros(Float64, mfcc_coeffs, time_length)
    s0 = sqrt(1 / time_length)
    s1 = sqrt(2 / time_length)
    piCCast = 2 * pi / (2 * time_length)

    matrix[1, :] .= s0
    for k in 1:time_length, n in 2:mfcc_coeffs
        matrix[n, k] = s1 * cos(piCCast * (n - 1) * (k - 0.5))
    end

    matrix
end

function _create_WArr(stft_length::Int64)
    cos_win = Float64[]
    sin_win = Float64[]

    length = 2 * stft_length
    for i in 1:stft_length
        push!(cos_win, cos(pi * i / length))
        push!(sin_win, -sin(pi * i / length))
    end

    return cos_win, sin_win
end

function cepstral_coefficients(
    mel_spec::AbstractMatrix{T},
    stft_length::Int64,
    n_coeffs::Int64,
    rectification::Symbol
) where {T<:AbstractFloat}

    time_length = size(mel_spec, 1)
    # Rectify
    if (rectification == :log)
        mel_spec[mel_spec.==0] .= floatmin(Float64)
        mel_spec = log10.(mel_spec)
    end

    # _fftObj_fft(fftObj,_realArr1 = mel_spec,_imageArr1 = ?,dataArr2,_imageArr2, 0);
    # length = fft length

    cos_win, sin_win = _create_WArr(stft_length)

    # Design DCT matrix
    DCTmatrix = create_DCT_matrix(n_coeffs, time_length)
    # Apply DCT matrix
    coeffs = DCTmatrix * mel_spec

    coeffs'
end # function cepstral_coefficients

## Funzione MFCC

c = cepstral_coefficients(data.mel_spectrogram', setup.stft.stft_length, setup.mfcc_coeffs, setup.rectification)

# c = cepstralCoefficients(data.mel_spectrogram', setup.mfcc_coeffs, setup.rectification)