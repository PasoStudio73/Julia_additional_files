using DSP
using FFTW
using Parameters
using PyCall

include("../src/windowing/windowing.jl")
include("../src/windowing/windows.jl")
include("../src/signalDataStructure.jl")
include("../src/fft/fft.jl")
include("../src/fft/lin.jl")
include("../src/fft/mel.jl")
include("../src/fft/spectral.jl")
include("../src/fft/f0.jl")

af = pyimport("audioflux")
librosa = pyimport("librosa")

sr_src = 16000
x, sr = librosa.load("/home/riccardopasini/Documents/Aclai/Datasets/SpcDS/SpcDS_gender_1000_60_100/WavFiles/common_voice_en_23616312.wav", sr=sr_src, mono=true)
FFTLength = 256
mel_num = 26

function createDCTmatrix(
    numCoeffs::Int64,
    numFilters::Int64,
    DT::DataType
)
    # create DCT matrix
    N = convert(DT, numCoeffs)
    K = numFilters
    matrix = zeros(DT, Int(N), numFilters)
    A = sqrt(1 / K)
    B = sqrt(2 / K)
    C = 2 * K
    piCCast = convert(DT, 2 * pi / C)

    # matrix[1, :] .= A
    # for k in 1:K
    #     for n in 1:Int(N)
    #         # matrix[n, k] = B * cos(piCCast * (n - 1) * (k - 0.5))
    #         matrix[n, k] = cos(pi * (n - 1) * (k - 0.5) / K)
    #     end
    # end

    for k in 1:K
        for n in 1:Int(N)
            # matrix[n, k] = B * cos(piCCast * (n - 1) * (k - 0.5))
            matrix[n, k] = cos(pi * (k - 1) * (n - 0.5) / K)
            println(matrix[n, k], " con k=", k, " n=", n, " K=", K)
        end
    end

    return matrix
end

# setup and data structures definition
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
    log_energy_source=:mfcc,
    log_energy_pos=:replace,
    delta_window_length=9,
    delta_matrix=:transposed,

    # spectral
    spectral_spectrum=:linear
)

# convert to Float64
x = Float64.(x)

data = signal_data(
    x=x
)

takeFFT(data, setup)
# lin_spectrogram(data, setup)
mel_spectrogram(data, setup)

S = data.mel_spectrogram'
S = ones(Float64, 26, 100)
S .= 10.0
numCoeffs = 13
rectification = :log

# function

DT = eltype(S)
# Rectify
if (rectification == :log)
    amin = floatmin(DT)
    S[S.==0] .= amin
    S = log10.(S)
end
# Reshape spectrogram to matrix for vectorized matrix multiplication
L, M = size(S)
N = 1
S = reshape(S, L, M * N)
# Design DCT matrix
DCTmatrix = createDCTmatrix(numCoeffs, L, DT)
# Apply DCT matrix
coeffs = DCTmatrix * S

A = sqrt(1 / L)
B = sqrt(2 / L)

coeffs[1, :] *= A
coeffs[(2:end), :] *= B

return permutedims(coeffs, [2 1])

coeffs'