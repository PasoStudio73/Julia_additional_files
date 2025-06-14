using Revise, Audio911, BenchmarkTools
using DSP, StaticArrays, SplitApplyCombine
using FFTW, LinearAlgebra, StatsBase, IterTools

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

audio = load_audio(source=wavfile, sr=16000)

# ---------------------------------------------------------------------------- #
#                                 stft structures                              #
# ---------------------------------------------------------------------------- #
# global const NORM_FUNCS = Dict(:power => x -> @. real(x * conj(x)), :magnitude => x -> @. abs(x))

@kwdef struct WindowFunctions
    hann::Function = x -> 0.5 * (1 + cospi(2x))
    hamming::Function = x -> 0.54 - 0.46 * cospi(2x)
    rect::Function = x -> 1.0
end

function get_window(wintype::Symbol, nwin::Int, winperiod::Bool)
    nwin == 1 && return [1.0]
    winfunc = getproperty(WindowFunctions(), wintype)
    winperiod && return (winfunc(x) for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1])
    (winfunc(x) for x in range(-0.5, stop=0.5, length=nwin))
end

function get_frames(x::AbstractVector{T}, nwin::Int, noverlap::Int) where T<:AbstractFloat
    nhop = nwin - noverlap
    nhops = div(length(x) - nwin, nhop) + 1
    @views (view(x, i:i+nwin-1) for i in 1:nhop:(nhops-1)*nhop+1)
end

function get_wframes(frames::Base.Generator, window::Base.Generator)
    win = collect(window)
    @fastmath @inbounds (i .* win for i in frames)
end

@inline function a911_stft(
    audio::Audio;
    nfft::Int = prevpow(2, audio.sr ÷ 30),
    nwin::Int = nfft,
    noverlap::Int = round(Int, nfft * 0.5),
    wintype::Symbol = :hann,
    winperiod::Bool = true,
    norm::Symbol = :power, # :none, :power, :magnitude
)
    (0 ≤ noverlap < nwin) || throw(DomainError((; noverlap, nwin), "noverlap must be between zero and nwin"))
    nfft >= nwin || throw(DomainError((; nfft, nwin), "nfft must be >= nwin"))
    @assert haskey(NORM_FUNCS, norm) "Unknown spectrum_type: $norm."

    window = get_window(wintype, nwin, winperiod)
    frames = get_frames(audio.data, nwin, noverlap)
    wframes = get_wframes(frames, window)

    plan = plan_rfft(1:nwin)
    combinedims(first.(NORM_FUNCS[norm](plan * frame) for frame in wframes))
end

######################################################################################################################
# s1 = DSP.stft(audio.data, 512, 256; onesided=true, nfft=512, fs=16000, window=hanning);
s2 = a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=false, norm=:power);

# @btime DSP.stft(audio.data, 512, 256; onesided=true, nfft=512, fs=16000, window=hanning);
@btime a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=false, norm=:power);
# @btime get_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=(:hann, :periodic))
;

