using Revise, Audio911, BenchmarkTools
using DSP, StaticArrays, SplitApplyCombine
using FFTW, LinearAlgebra, StatsBase, IterTools

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

audio = load_audio(source=wavfile, sr=16000)


# ---------------------------------------------------------------------------- #
#                                 stft structures                              #
# ---------------------------------------------------------------------------- #
@kwdef struct WindowFunctions
    hann::Function = x -> 0.5 * (1 + cospi(2x))
    hamming::Function = x -> 0.54 - 0.46 * cospi(2x)
    rect::Function = x -> 1.0
end

# ---------------------------------------------------------------------------- #
#                              audio windowed frames                           #
# ---------------------------------------------------------------------------- #
struct WFrames{T<:AbstractVector, S<:AbstractFloat, W} <: AbstractVector{Vector{S}}
    x::T
    buf::Vector{S}
    nwin::Int
    noverlap::Int
    nhop::Int
    nhops::Int
    window::W

    function WFrames{Ti,Si,Wi}(x, nfft, nwin, noverlap, window; buffer::Vector{Si}=zeros(Si, max(nfft, 0))) where {Ti<:AbstractVector,Si,Wi}
        (0 ≤ noverlap < nwin) || throw(DomainError((; noverlap, nwin), "noverlap must be between zero and nwin"))
        nfft >= nwin || throw(DomainError((; nfft, nwin), "nfft must be >= nwin"))
        length(buffer) == nfft || throw(ArgumentError("buffer length ($(length(buffer))) must equal `nfft` ($nfft)"))

        nhop = nwin - noverlap
        nhops = div(length(x) - nwin, nhop) + 1

        new{Ti,Si,Wi}(x, buffer, nwin, noverlap, nhop, nhops, window)
    end
end

WFrames(x::T, nfft::Int, nwin::Int, noverlap::Int, window::W) where {S<:AbstractFloat, T<:AbstractVector{S}, W} = WFrames{T,S,W}(x, nfft, nwin, noverlap, window)

function get_window(wintype::Symbol, nwin::Int; winperiod::Bool = true)::Vector{Float64}
    nwin == 1 && return [1.0]
    winfunc::Function = getproperty(WindowFunctions(), wintype)
    winperiod && return collect(winfunc(x)::Float64 for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1])
    collect(winfunc(x)::Float64 for x in range(-0.5, stop=0.5, length=nwin))
end

function get_frame(x::AbstractVector{<:AbstractFloat}, i::Int, nwin::Int, noverlap::Int)::Vector{Float64}
    offset = (i - 1) * (nwin - noverlap) + 1
    view(x, offset:offset+nwin-1)
end
get_frame(x::WFrames, i::Int) = get_frame(x.x, i, x.nwin, x.noverlap)
get_frames(x::WFrames) = collect(get_frame(x.x, i, x.nwin, x.noverlap) for i in x.nhops)

function Base.getindex(x::WFrames{T,S,Nothing} where {T<:AbstractVector,S}, i::Int)
    @boundscheck (1 <= i <= x.nwin) || throw(BoundsError(x, i))
    copyto!(x.buf, 1, x.x, (i - 1) * (x.nwin - x.noverlap) + firstindex(x.x), x.n)
end
function Base.getindex(x::WFrames, i::Int)
    @boundscheck (1 <= i <= x.nwin) || throw(BoundsError(x, i))
    # window = x.window
    # offset = (i - 1) * (x.nwin - x.noverlap) + 1
    # for i = 1:x.nwin
    #     @inbounds x.buf[i] = x.x[offset+i] * window[i]
    # end
    x.buf = @inbounds get_frame(x, i)
    x.buf
end

Base.IndexStyle(::WFrames) = IndexLinear()
Base.iterate(x::WFrames, i::Int = 1) = (i > x.nhops ? nothing : (x[i], i+1))
Base.size(x::WFrames) = (x.nhops,)

function get_wframes(x::AbstractVector{<:AbstractFloat}, nfft::Int, nwin::Int, noverlap::Int, wintype, winperiod)
    window = get_window(wintype, nwin; winperiod=winperiod)
    WFrames(x, nfft, nwin, noverlap, window)
end

Base.collect(x::WFrames) = collect(copy(a) for a in x)





@inline function a911_stft(
    audio::Audio;
    nfft::Int = prevpow(2, audio.sr ÷ 30),
    nwin::Int = nfft,
    noverlap::Int = round(Int, nfft * 0.5),
    wintype::Symbol = :hann,
    winperiod::Bool = true,
    norm::Symbol = :power, # :none, :power, :magnitude
)
    # @assert haskey(NORM_FUNCS, norm) "Unknown spectrum_type: $norm."

    # window = get_window(wintype, nwin; winperiod=winperiod)
    # frames = get_frames(audio.data, nwin, noverlap)
    sig_split = get_wframes(audio.data, nfft, nwin, noverlap, wintype, winperiod)
end

######################################################################################################################

# s2 = a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=false, norm=:power);

# @btime $s1 = DSP.stft(audio.data, 512, 256, nothing; onesided=true, nfft=512, fs=16000, window=hanning);
@btime s2 = a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=false, norm=:power);

# @benchmark $s1 = dsp_stft(audio.data, 512, 256, nothing; onesided=true, nfft=512, fs=16000, window=hanning)
# @benchmark $s2 = a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=false, norm=:power)

