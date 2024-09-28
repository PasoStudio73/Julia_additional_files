using SpecialFunctions, Roots
using Statistics
using FFTW
using Parameters
using Plots
using SparseArrays

using Revise, Audio911, BenchmarkTools

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr_src = 16000
rack = load_audio(wavfile, sr_src)

wavelet = :morlet # :morse, :morlet, :bump
morse_params = (3, 60)
vpo = 10
freq_range = (50, 8000)
full_process = true

sr = rack.audio.sr
x = rack.audio.data
x_length = size(rack.audio.data, 1)

function get_freq_cutoff_morlet(cutoff::Int64, cf::Real)
    alpha = 0.02 * cutoff
    omax = √1500 + cf
    psihat(x) = alpha - 2 * exp(-(x - cf)^2 / 2)
    
    psihat(cf) > 0 ? omax : find_zero(psihat, (cf, omax))
end

### frequency_grid
ga, be = (morse_params[1], morse_params[2] / morse_params[1])
omega = range(0, step=(2π / x_length), length=floor(Int, x_length / 2) + 1)
wrange = freq_range .* (2π / sr)
# scales, center_freq = cwt_scales(wavelet, x_length, ga, be, vpo, wrange)

### cwt_scales
cutoff = wavelet == :morse ? 50 : 10

if wavelet == :morse
    center_freq, _ = morsepeakfreq(ga, be)
    _, _, _, sigmaT, _ = morseproperties(ga, be)

    omegaC = get_freq_cutoff_morse(cutoff, center_freq, ga, be)

elseif wavelet == :morlet
    center_freq = 6
    sigmaT = sqrt(2)

    omegaC = get_freq_cutoff_morlet(cutoff, center_freq)

elseif wavelet == :bump
    center_freq = 5
    # measured standard deviation of bump wavelet
    sigmaT = 5.847705

    omegaC = get_freq_cutoff_bump(cutoff, center_freq)
else
    error("Wavelet $wavelet not supported.")
end

wfreq = (center_freq / wrange[2], center_freq / wrange[1])
n_octaves = log2(wfreq[2] / wfreq[1])
a0 = 2^(1 / vpo)

scales = wfreq[1] * a0 .^ (0:(vpo * n_octaves))

somega = scales .* omega'

### normale
@btime begin
    somega = scales .* omega'

    if wavelet == :morse
        absomega = abs.(somega)
        if ga == 3
            powscales = absomega .* absomega .* absomega
        else
            powscales = absomega .^ ga
        end
        factor = exp(-be * log(center_freq) + center_freq^ga)
        fbank = 2 * factor * exp.(be .* log.(absomega) - powscales) .* (somega .> 0)

    elseif wavelet == :morlet
        fc = 6
        mul = 2
        squareterm = (somega .- fc) .* (somega .- fc)
        gaussexp = -squareterm ./ 2
        expnt = gaussexp .* (somega .> 0)
        fbank = mul * exp.(expnt) .* (somega .> 0)

    elseif wavelet == :bump
        fc = 5
        sigma = 0.6
        w = (somega .- fc) ./ sigma
        absw2 = w .* w
        expnt = -1 ./ (1 .- absw2)
        daughter = 2 * exp(1) * exp.(expnt) .* (abs.(w) .< 1 .- eps(1.0))
        daughter[isnan.(daughter)] .= 0
        fbank = daughter

    else
        error("Wavelet $wavelet not supported.")
    end

    cwt_freq = (center_freq ./ scales) / (2 * pi) .* sr

    ###
    full_process ? fbank = hcat(fbank, zeros(size(fbank[:,2:end]))) : nothing
    ifft(fft(x)[1:size(fbank, 2)]' .* fbank, 2)
end

### sparsa
@btime begin
    somega = scales .* omega'

    if wavelet == :morse
        absomega = abs.(somega)
        if ga == 3
            powscales = absomega .* absomega .* absomega
        else
            powscales = absomega .^ ga
        end
        factor = exp(-be * log(center_freq) + center_freq^ga)
        fbank = 2 * factor * exp.(be .* log.(absomega) - powscales) .* (somega .> 0)

    elseif wavelet == :morlet
        fc, mul = 6, 2
        squareterm = (somega .- fc) .^ 2
        expnt = -squareterm ./ 2 .* (somega .> 0)
        fbank = sparse(mul * exp.(expnt) .* (somega .> 0))

    elseif wavelet == :bump
        fc = 5
        sigma = 0.6
        w = (somega .- fc) ./ sigma
        absw2 = w .* w
        expnt = -1 ./ (1 .- absw2)
        daughter = 2 * exp(1) * exp.(expnt) .* (abs.(w) .< 1 .- eps(1.0))
        daughter[isnan.(daughter)] .= 0
        fbank = daughter

    else
        error("Wavelet $wavelet not supported.")
    end

    cwt_freq = (center_freq ./ scales) / (2 * pi) .* sr

    ###
    full_process ? fbank = hcat(fbank, zeros(size(fbank[:,2:end]))) : nothing
    ifft(fft(x)[1:size(fbank, 2)]' .* collect(fbank), 2)
end




# fourier transform of input
xposdft = fft(y)
# obtain the CWT in the Fourier domain
cfsposdft = xposdft[1:size(fbank, 2)]' .* fbank
# cfsposdft = xposdft' .* hcat(fbank, zeros(size(fbank[:,2:end])))
# invert to obtain wavelet coefficients
cfs = ifft(cfsposdft, 2)

# function frequency_grid(
#     sr::Int64,
#     x_length::Int64,
#     signal_pad::Int64
# )
#     n = x_length + 2 * signal_pad

#     omega = [1:floor(Int, n / 2)...] .* ((2 * pi) / n)
#     omega = vcat(0.0, omega, -omega[floor(Int, (n - 1) / 2):-1:1])
#     frequencies = sr * omega ./ (2 * pi)

#     return omega, frequencies
# end

c1 = X.cwt_fb.fbank
f1 = X.cwt_fb.freq
c2 = X.mel_fb.fbank
f2 = X.mel_fb.freq

fbank = X.mel_fb.fbank
# fourier transform of input
xposdft = fft(x.audio.data)
# obtain the CWT in the Fourier domain
cfsposdft = xposdft[1:size(fbank, 1)]' .* fbank'
cfs = ifft(cfsposdft, 2)

heatmap(abs.(cfs))

fbank = fbank'
# cfsposdft = xposdft' .* hcat(fbank, zeros(size(fbank[:,2:end])))
# invert to obtain wavelet coefficients
