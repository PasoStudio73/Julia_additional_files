using Revise, Audio911, BenchmarkTools
using DSP, StaticArrays, SplitApplyCombine
using FFTW, LinearAlgebra, StatsBase, SIMD

import Base: Generator

import SplitApplyCombine
using SplitApplyCombine: _inneraxes, _combine_tuples, slice_inds
import SplitApplyCombine: combinedims, _combinedims

function _combine_generator_dims(a::Base.Generator, od::Val{outer_dims}, firstinner::AbstractVector) where {outer_dims}
    outeraxes = axes(a)
    inneraxes = SplitApplyCombine._inneraxes(a)
    ndims_total = length(outeraxes) + length(inneraxes)
    newaxes = SplitApplyCombine._combine_tuples(ndims_total, outer_dims, outeraxes, inneraxes)

    T = eltype(firstinner)
    out = Array{T}(undef, length.(newaxes)...)
    for (j,v) in zip(CartesianIndices(outeraxes),a)
        I = SplitApplyCombine.slice_inds(j, od, Val(ndims_total))
        view(out, I...) .= v
    end
    return out
end

function _combine_generator_dims(a::Base.Generator, od::Val{outer_dims}) where {outer_dims}
    firstinner = first(iterate(a))
    return _combine_generator_dims(a, od, firstinner)
end

@inline function combine_generator_dims(a::Base.Generator)
    N = length(SplitApplyCombine.axes(a))
    M = length(SplitApplyCombine._inneraxes(a))
    _combine_generator_dims(a, Val(ntuple(i -> N + i, M)))
end

# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

audio = load_audio(source=wavfile, sr=16000)

########################################################################################################################################
struct ArroySplit{T<:AbstractVector,S,W} <: AbstractVector{Vector{S}}
    s::T
    buf::Vector{S}
    n::Int
    noverlap::Int
    window::W
    k::Int

    function ArroySplit{Ti,Si,Wi}(s, n, noverlap, nfft, window) where {Ti<:AbstractVector,Si,Wi}
        buffer::Vector{Si}=zeros(Si, max(nfft, 0))
        # n = noverlap is a problem - the algorithm will not terminate.
        (0 ≤ noverlap < n) || throw(DomainError((; noverlap, n), "noverlap must be between zero and n"))
        nfft >= n || throw(DomainError((; nfft, n), "nfft must be >= n"))
        length(buffer) == nfft ||
            throw(ArgumentError("buffer length ($(length(buffer))) must equal `nfft` ($nfft)"))

        new{Ti,Si,Wi}(s, buffer, n, noverlap, window, length(s) >= n ? div((length(s) - n),
            n - noverlap) + 1 : 0)
    end

end
ArroySplit(s::T, n, noverlap, nfft, window::W; kwargs...) where {S,T<:AbstractVector{S},W} =
ArroySplit{T,fftintype(S),W}(s, n, noverlap, nfft, window; kwargs...)

# function Base.getindex(x::ArroySplit{T,S,Nothing} where {T<:AbstractVector,S}, i::Int)
#     @boundscheck (1 <= i <= x.k) || throw(BoundsError(x, i))
#     copyto!(x.buf, 1, x.s, (i - 1) * (x.n - x.noverlap) + firstindex(x.s), x.n)
# end
function Base.getindex(x::ArroySplit, i::Int)
    @boundscheck (1 <= i <= x.k) || throw(BoundsError(x, i))
    
    offset = (i - 1) * (x.n - x.noverlap) + firstindex(x.s) - 1
    window = x.window
    for i = 1:x.n
        @inbounds x.buf[i] = x.s[offset+i] * window[i]
    end
    x.buf
end

Base.IndexStyle(::ArroySplit) = IndexLinear()
Base.iterate(x::ArroySplit, i::Int = 1) = (i > x.k ? nothing : (x[i], i+1))
Base.size(x::ArroySplit) = (x.k, )

arroysplit(s, n, noverlap, nfft=n, window=nothing; kwargs...) = ArroySplit(s, n, noverlap, nfft, window; kwargs...)

## Make collect() return the correct split arrays rather than repeats of the last computed copy
Base.collect(x::ArroySplit) = collect(copy(a) for a in x)

struct PSDOnly end
stfttype(T::Type, ::PSDOnly) = fftabs2type(T)
stfttype(T::Type, ::Nothing) = fftouttype(T)

compute_window(::Nothing, n::Int) = (nothing, n)
function compute_window(window::Function, n::Int)
    win = window(n)
    norm2 = sum(abs2, win)
    (win, norm2)
end

function dsp_stft(s::AbstractVector{T}, n::Int=length(s) >> 3, noverlap::Int=n >> 1,
    psdonly::Union{Nothing,PSDOnly}=nothing;
    onesided::Bool=T <: Real, nfft::Int=nextfastfft(n), fs::Real=1,
    window::Union{Function,AbstractVector,Nothing}=nothing) where {T}

    onesided && T <: Complex && throw(ArgumentError("cannot compute one-sided FFT of a complex signal"))

    win, norm2 = compute_window(window, n)
    # sig_split = arraysplit(s, n, noverlap, nfft, win)
    # return win, sig_split
    # collect(sig_split)
end

#########################################################################################################################################

@kwdef struct WindowFunctions
    hann::Function = x -> 0.5 * (1 + cospi(2x))
    hamming::Function = x -> 0.54 - 0.46 * cospi(2x)
    rect::Function = x -> 1.0
    wframes::Function = (x,y) -> x * y
end

# struct Window
#     window::Generator

#     function Window(wintype::Symbol, nwin::Int; winperiod::Bool = true)
#         nwin == 1 && return new(Base.Generator(Returns(1.0), 1))
#         winfunc = getproperty(WindowFunctions(), wintype)
#         winperiod && return new(winfunc(x) for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1])
#         new(winfunc(x) for x in range(-0.5, stop=0.5, length=nwin))
#     end
# end
# get_window(wintype::Symbol, nwin::Int; kwargs...) = Window(wintype, nwin; kwargs...)

abstract type AbstractWindow end

struct Window <: AbstractWindow window::Vector{Float64} end

# struct Window <: AbstractWindow 
#     winfunc::Function
#     window::Vector{Float64}
#     nwin::Int
#     nhop::Int
#     nhops::Int
#     winperiod::Bool

#     function Window(xlength::Int, wintype::Symbol, nfft::Int, nwin::Int, noverlap::Int; winperiod::Bool = true)
#         noverlap < nwin || throw(ArgumentError("Overlap length must be smaller than nwin: $nwin."))
#         nwin ≤ nfft || throw(ArgumentError("FFT window size smaller than actual window size is highly discouraged."))

#         window::Vector{Float64}=zeros(Float64, nwin)
#         nhop = nwin - noverlap
#         new(getproperty(WindowFunctions(), wintype), window, nwin, nhop, div(length(xlength) - nwin, nhop) + 1, winperiod)
#     end
# end

# function Base.getindex(w::Window, i::Int)
#     @boundscheck (1 <= i <= w.nwin) || throw(BoundsError(w, i))
#     if w.nwin == 1
#         @inbounds w.window = [1.0]
#     elseif w.winperiod 
#         for x in range(-0.5, stop=0.5, length=w.nwin+1)
#         @inbounds w.window[x] = w.winfunc(x)
#         end
#     else
#         @inbounds w.window = collect(w.winfunc(x) for x in range(-0.5, stop=0.5, length=w.nwin))
#     end
# end
# Base.IndexStyle(::Window) = IndexLinear()
# Base.iterate(w::Window, i::Int = 1) = (i > w.nwin ? nothing : (w[i], i+1))
# Base.size(w::Window) = (w.nwin, )
# Base.collect(w::Window) = collect(copy(i) for i in w)

# get_window(args...; kwargs...) = Window(args...; kwargs...)

function get_window(wintype::Symbol, nwin::Int; winperiod::Bool = true)
    nwin == 1 && return Window([1.0])
    winfunc::Function = getproperty(WindowFunctions(), wintype)
    winperiod && return Window(@inbounds collect(winfunc(x) for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1]))
    Window(@inbounds collect(winfunc(x) for x in range(-0.5, stop=0.5, length=nwin)))
    # winperiod && return Window(map(winfunc, range(-0.5, stop=0.5, length=nwin+1)[1:end-1]))
    # Window(map(winfunc, range(-0.5, stop=0.5, length=nwin)))
    # winperiod && return Window(@inbounds collect(winfunc(x) for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1]))
    # Window(@inbounds collect(winfunc(x) for x in range(-0.5, stop=0.5, length=nwin)))
end

function get_frames(x::AbstractVector{T}, nwin::Int, nhop::Int, nhops::Int) where T<:AbstractFloat
    (view(x, i:i+nwin-1) for i in range(1, step=nhop, length=nhops))
end

function get_wframes(f::Generator, w::Window)
    # win = collect(w.window)
    @fastmath @inbounds collect(i .* w.window for i in collect(f))
end

# 218.787 μs (375 allocations: 1.51 MiB)

@inline function a911_stft(
    audio::Audio;
    nfft::Int = prevpow(2, audio.sr ÷ 30),
    nwin::Int = nfft,
    noverlap::Int = round(Int, nfft * 0.5),
    wintype::Symbol = :hann,
    winperiod::Bool = true,
    norm::Symbol = :power, # :none, :power, :magnitude
)
    noverlap < nwin || throw(ArgumentError("Overlap length must be smaller than nwin: $nwin."))
    nwin ≤ nfft || throw(ArgumentError("FFT window size smaller than actual window size is highly discouraged."))
    # @assert haskey(NORM_FUNCS, norm) "Unknown spectrum_type: $norm."

    nhop = nwin - noverlap
    nhops = div(length(audio.data) - nwin, nhop) + 1

    window = get_window(wintype, nwin; winperiod=winperiod)
    frames = get_frames(audio.data, nwin, nhop, nhops)
    wframes = get_wframes(frames, window)

    # wframes = get_frames(audio.data, winsetup.win, nwin, noverlap)
    # collect(winsetup.wframes)

    # winsetup
    # combinedims(collect(frames))
    # combinedims(frames)

    # nout = (nfft >> 1)+1
    # spec = zeros(T, nout, size(wframes, 2))
    # tmp = Vector{ComplexF64}(undef, nout)

    # plan = plan_rfft(1:nfft)
    # offset = 0

    # @inbounds @simd for i in eachcol(wframes)
    #     mul!(tmp, plan, i)
    #     copyto!(spec, offset+1, NORM_FUNCS[norm](tmp), 1, nout)
    #     offset += nout
    # end
end

######################################################################################################################

s1 = dsp_stft(audio.data, 512, 256, nothing; onesided=true, nfft=512, fs=16000, window=hanning);
s2 = a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=true, norm=:power);

@btime s1 = dsp_stft(audio.data, 512, 256, nothing; onesided=true, nfft=512, fs=16000, window=hanning);
@btime s2 = a911_stft(audio; nfft=512, nwin=512, noverlap=256, wintype=:hann, winperiod=true, norm=:power);

