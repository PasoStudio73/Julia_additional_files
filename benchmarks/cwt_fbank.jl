using Pkg
Pkg.activate("/home/paso/Documents/Aclai/audio-rules2024")
using BenchmarkTools, Audio911
using StaticArrays, SparseArrays, LinearAlgebra, Base.Threads

using SpecialFunctions, Roots
using Statistics

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

# --- audio ------------------------------------------------------------------ #
sr = 16000
audio = load_audio(file=wavfile, sr=sr, norm=true);
x_length = size(audio.data, 1)

# ---------------------------------------------------------------------------- #
#                                     utils                                    #
# ---------------------------------------------------------------------------- #
function morsepeakfreq(ga::Real, be::Real)
    # peak frequency for 0-th order Morse wavelet is $(\frac{\beta}{\gamma})^{1/\gamma}$
    peakAF = exp(1 / ga * (log(be) - log(ga)))
    # obtain the peak frequency in cyclical frequency
    peakCF = peakAF / 2π

    return peakAF, peakCF
end

function morseproperties(ga::Real, be::Real)
    width = √(ga * be)
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

    sigmaF = √(exp(logsigo1) - exp(logsigo2))

    ra = morse_gb - 2 * morse_loga(be - 1)           + morse_loga(2 * (be - 1))
    rb = morse_gb - 2 * morse_loga(be - 1 + ga)      + morse_loga(2 * (be - 1 + ga))
    rc = morse_gb - 2 * morse_loga(be - 1 + ga ./ 2) + morse_loga(2 * (be - 1 + ga ./ 2))

    logsig2a = ra + lg_sig2 + 2 * lb + loggamma((2 * (be - 1) + 1) / ga) - lg_sig3
    logsig2b = rb + lg_sig2 + 2 * lg + loggamma((2 * (be - 1 + ga) + 1) / ga) - lg_sig3
    logsig2c = rc + lg_sig2 + log(2) + lb + lg + loggamma((2 * (be - 1 + ga ./ 2) + 1) / ga) - lg_sig3

    sigmaT = √(exp(logsig2a) + exp(logsig2b) - exp(logsig2c))

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
    ga::Real,
    be::Real,
    vpo::Int64,
    freq_range::Tuple{Int64, Int64},
    sr::Int64,
    x_length::Int64
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

    minfreq = sigmaT * center_freq * sr / (π * x_length)

    if freq_range[1] < minfreq
        freq_range = (minfreq, freq_range[2])
    end

    wrange = @inbounds (@. sr / (2π * freq_range))
    wfreq = (center_freq * wrange[2], center_freq * wrange[1])

    a0 = 2^(1 / vpo)
    n_octaves = log2(wfreq[2] / wfreq[1])
    scales = @views wfreq[1] * a0 .^ (0:(vpo * n_octaves))

    return scales, center_freq
end

# ---------------------------------------------------------------------------- #
#                       continuous wavelets filterbank                         #
# ---------------------------------------------------------------------------- #
cwt_fb = CwtFbank(sr=sr, wavelet = :morse)

function cwt_bench(cwt_fb, x_length)
    ga, be = (cwt_fb.morse_params[1], cwt_fb.morse_params[2] / cwt_fb.morse_params[1])

    omega = range(0, step=(2π / x_length), length=floor(Int, x_length / 2) + 1)
    scales, center_freq = cwt_scales(cwt_fb.wavelet, ga, be, cwt_fb.vpo, cwt_fb.freq_range, cwt_fb.sr, x_length)

    somega = scales * omega'


    if cwt_fb.wavelet == :morse
        absomega = @inbounds abs.(somega)
        powscales = ga == 3 ? absomega .^ 3 : absomega .^ ga
        factor = exp(-be * log(center_freq) + center_freq^ga)
        cwt_fb.fbank = @views sparse(@. 2 * factor * exp(be * log(absomega) - powscales) * (somega > 0))

    elseif cwt_fb.wavelet == :morlet
        fc, mul = 6, 2
        squareterm = @. (somega - fc) ^ 2
        expnt = @. -squareterm / 2 * (somega > 0)
        cwt_fb.fbank = @views sparse(@. mul * exp(expnt) * (somega > 0))

    elseif cwt_fb.wavelet == :bump
        fc, sigma = 5, 0.6
        w = @. (somega - fc) / sigma
        absw2 = w .^ 2
        expnt = @. -1 / (1 - absw2)
        daughter = sparse(@. 2 * exp(1) * exp(expnt) * (abs(w) < 1 - eps(1.0)))
        # daughter[isnan.(daughter)] .= 0
        cwt_fb.fbank = @views (@. daughter[isnan(daughter)] = 0)A

    else
    #     error("Wavelet $cwt_fb.wavelet not supported.")
    end

    cwt_fb.freq = @. (center_freq / scales) / (2π) * cwt_fb.sr
    cwt_fb.fbank = hcat(cwt_fb.fbank, spzeros(eltype(cwt_fb.fbank), size(cwt_fb.fbank, 1), x_length - size(cwt_fb.fbank, 2)))
end
;

@btime cwt_bench(cwt_fb, x_length);

# sparse
# 92.833 ms (86 allocations: 184.84 MiB)

@btime somega = scales * omega';

@btime somega = map((s) -> s * omega, scales);

@btime somega = stack(collect.(map((s) -> s * omega, scales)), dims=1);

@btime somega = reshape(reinterpret(Float64, Float64.(Iterators.flatten(collect.(map((s) -> s * omega, scales))))), (size(s2,1), :));


@btime begin
    s1 = scales * omega'
    a1 = abs.(s1);
end

@btime begin
    s2 = map((s) -> s * omega, scales)
    ao = abs.(omega)
    a2 = map((s) -> abs(s) * ao, scales);
end

s1 = scales * omega'
a1 = abs.(s1);

s2 = map((s) -> s * omega, scales)
ao = abs.(omega)
a2 = map((s) -> abs(s) * ao, scales)