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

sr = 16000
audio = load_audio(file=wavfile, sr=sr)
audio.data = audio.data[1:34564]

wavelet = :bump # :morse, :morlet, :bump
morse_params = (3, 60)
vpo = 10
freq_range = (80, 8000)
full_process = true

sr = audio.sr
x = audio.data
x_length = size(x, 1)

# ---------------------------------------------------------------------------- #
#                                     utils                                    #
# ---------------------------------------------------------------------------- #
function morsepeakfreq(ga::Real, be::Real)
    # peak frequency for 0-th order Morse wavelet is $(\frac{\beta}{\gamma})^{1/\gamma}$
    peakAF = exp(1 / ga * (log(be) - log(ga)))
    # obtain the peak frequency in cyclical frequency
    peakCF = peakAF / (2 * pi)

    return peakAF, peakCF
end

function morseproperties(ga::Real, be::Real)
    width = sqrt(ga * be)
    skew = (ga - 3) / width
    kurt = 3 - skew .^ 2 - (2 / width^2)
    
    lg, lb = log(ga), log(be)
    morse_loga = (be) -> be / ga * (1 + lg - log(be))
    morse_gb = 2 * morse_loga(be) - morse_loga(2 * be)

    lg_sig1 = 2 / ga * log(ga / (2 * be))
    lg_sig2 = 2 / ga * log(be / ga)
    lg_sig3 = loggamma((2 * be + 1) / ga)

    logsigo1 = lg_sig1 + loggamma((2 * be + 1 + 2) / ga) - lg_sig3
    logsigo2 = lg_sig1 + 2 * loggamma((2 * be + 2) / ga) - 2 * lg_sig3

    sigmaF = sqrt(exp(logsigo1) - exp(logsigo2))

    ra = morse_gb - 2 * morse_loga(be - 1)           + morse_loga(2 * (be - 1))
    rb = morse_gb - 2 * morse_loga(be - 1 + ga)      + morse_loga(2 * (be - 1 + ga))
    rc = morse_gb - 2 * morse_loga(be - 1 + ga ./ 2) + morse_loga(2 * (be - 1 + ga ./ 2))

    logsig2a = ra + lg_sig2 + 2 * lb + loggamma((2 * (be - 1) + 1) / ga) - lg_sig3
    logsig2b = rb + lg_sig2 + 2 * lg + loggamma((2 * (be - 1 + ga) + 1) / ga) - lg_sig3
    logsig2c = rc + lg_sig2 + log(2) + lb + lg + loggamma((2 * (be - 1 + ga ./ 2) + 1) / ga) - lg_sig3

    sigmaT = sqrt(exp(logsig2a) + exp(logsig2b) - exp(logsig2c))

    return width, skew, kurt, sigmaT, sigmaF
end

function get_freq_cutoff_morse(cutoff::Int64, cf::Real, ga::Real, be::Real)
    anorm = 2 * exp(be / ga * (1 + (log(ga) - log(be))))
    alpha = 2 * (cutoff / 100)

    psihat = x -> alpha - anorm * x .^ be * exp(-x .^ ga)

    omax = ((750) .^ (1 / ga))
    if psihat(cf) >= 0
        if psihat(omax) == psihat(cf)
            omegaC = omax
        else
            omegaC = cf
        end
    else
        omegaC = find_zero(psihat, (cf, omax))
    end
end

function get_freq_cutoff_morlet(cutoff::Int64, cf::Real)
    alpha = 0.02 * cutoff
    omax = √1500 + cf
    psihat(x) = alpha - 2 * exp(-(x - cf)^2 / 2)
    
    psihat(cf) > 0 ? omax : find_zero(psihat, (cf, omax))
end

function get_freq_cutoff_bump(cutoff::Int64, cf::Real)
    sigma = 0.6
    cutoff = cutoff / 100

    if cutoff < 10 * eps(0.0)
        omegaC = cf + sigma - 10 * eps(cf + sigma)
    else
        alpha = 2 * cutoff

        psihat = x -> 1 / (1 - x^2) + log(alpha) - log(2) - 1

        epsilon = find_zero(psihat, (0 + eps(0.0), 1 - eps(1.0)))
        omegaC = sigma * epsilon + cf
    end
end

# ---------------------------------------------------------------------------- #
#              construct the frequency grid for the wavelet DFT                #
# ---------------------------------------------------------------------------- #
function cwt_scales(
    wavelet::Symbol,
    x_length::Int64,
    ga::Real,
    be::Real,
    vpo::Int64,
    wrange::Tuple{Real, Real}
)
    cutoff = wavelet == :morse ? 50 : 10

    if wavelet == :morse
        center_freq, _ = morsepeakfreq(ga, be)
        _, _, _, sigmaT, _ = morseproperties(ga, be)

        omegaC = get_freq_cutoff_morse(cutoff, center_freq, ga, be)

    elseif wavelet == :morlet
        center_freq = 6
        sigmaT = √2

        omegaC = get_freq_cutoff_morlet(cutoff, center_freq)

    elseif wavelet == :bump
        center_freq = 5
        # measured standard deviation of bump wavelet
        sigmaT = 5.847705

        omegaC = get_freq_cutoff_bump(cutoff, center_freq)
    else
        error("Wavelet $wavelet not supported.")
    end

    # check frequency limits


    wfreq = (center_freq / wrange[2], center_freq / wrange[1])
    n_octaves = log2(wfreq[2] / wfreq[1])
    a0 = 2^(1 / vpo)

    return wfreq[1] * a0 .^ (0:(vpo * n_octaves)), center_freq
end

# ---------------------------------------------------------------------------- #
#                       continuous wavelets filterbank                         #
# ---------------------------------------------------------------------------- #
function _get_cwt_fb(;
    x_length::Int64,
    cwt_fb::CwtFbank,
)
    ga, be = (cwt_fb.morse_params[1], cwt_fb.morse_params[2] / cwt_fb.morse_params[1])

    omega = range(0, step=(2π / x_length), length=floor(Int, x_length / 2) + 1)
    wrange = cwt_fb.freq_range .* (2π / cwt_fb.sr)
    scales, center_freq = cwt_scales(cwt_fb.wavelet, ga, be, cwt_fb.vpo, wrange)

    somega = scales .* omega'

    if cwt_fb.wavelet == :morse
        absomega = abs.(somega)
        powscales = ga == 3 ? absomega .^ 3 : absomega .^ ga
        factor = exp(-be * log(center_freq) + center_freq^ga)
        cwt_fb.fbank = 2 * factor * exp.(be .* log.(absomega) - powscales) .* (somega .> 0)

    elseif cwt_fb.wavelet == :morlet
        fc, mul = 6, 2
        squareterm = (somega .- fc) .^ 2
        expnt = -squareterm ./ 2 .* (somega .> 0)
        cwt_fb.fbank = mul * exp.(expnt) .* (somega .> 0)

    elseif cwt_fb.wavelet == :bump
        fc, sigma = 5, 0.6
        w = (somega .- fc) ./ sigma
        absw2 = w .^ 2
        expnt = -1 ./ (1 .- absw2)
        daughter = 2 * exp(1) * exp.(expnt) .* (abs.(w) .< 1 .- eps(1.0))
        daughter[isnan.(daughter)] .= 0
        cwt_fb.fbank = daughter

    else
        error("Wavelet $cwt_fb.wavelet not supported.")
    end

    cwt_fb.freq = (center_freq ./ scales) / (2 * pi) .* cwt_fb.sr

    cwt_fb.full_proc ? cwt_fb.fbank = hcat(cwt_fb.fbank, zeros(size(cwt_fb.fbank[:,2:end]))) : nothing

    return cwt_fb
end

function get_cwt_fb(;
	audio::Audio,
	kwargs...
)
    _get_cwt_fb(; x_length=size(audio.data, 1), cwt_fb=CwtFbank(; sr=audio.sr, kwargs...))
end