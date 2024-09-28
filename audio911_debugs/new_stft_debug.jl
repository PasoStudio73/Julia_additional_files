using Revise, Audio911, BenchmarkTools
using DSP, StaticArrays, SplitApplyCombine
using FFTW, LinearAlgebra, StatsBase
using IterTools

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
    win = window(n)::Vector{Float64}
    norm2 = sum(abs2, win)
    (win, norm2)
end

function dsp_stft(s::AbstractVector{T}, n::Int=length(s) >> 3, noverlap::Int=n >> 1,
    psdonly::Union{Nothing,PSDOnly}=nothing;
    onesided::Bool=T <: Real, nfft::Int=nextfastfft(n), fs::Real=1,
    window::Union{Function,AbstractVector,Nothing}=nothing) where {T}

    onesided && T <: Complex && throw(ArgumentError("cannot compute one-sided FFT of a complex signal"))

    win, norm2 = compute_window(window, n)
    sig_split = arraysplit(s, n, noverlap, nfft, win)
    # return win, sig_split
    collect(sig_split)
end

#########################################################################################################################################

@kwdef struct WindowFunctions
    hann::Function = x -> 0.5 * (1 + cospi(2x))
    hamming::Function = x -> 0.54 - 0.46 * cospi(2x)
    rect::Function = x -> 1.0
end

struct Window
    window::Generator
    nwin::Int

    function Window(wintype::Symbol, nwin::Int; winperiod::Bool = true)
        nwin == 1 && return new(Base.Generator(Returns(1.0), 1), nwin)
        winfunc = getproperty(WindowFunctions(), wintype)
        winperiod && return new((winfunc(x) for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1]), nwin)
        new((winfunc(x) for x in range(-0.5, stop=0.5, length=nwin)), nwin)
    end
end


function Base.getindex(w::Window, i::Int)
    @boundscheck (1 <= i <= w.nwin) || throw(BoundsError(x, i))
    nth(w.window, i)
    # w.window
end
# Base.length(w::Window) = w.nwin
Base.iterate(w::Window, i::Int = 1) = (i > w.nwin ? nothing : (w[i], i+1))
Base.IndexStyle(::ArroySplit) = IndexLinear()

# Base.size(x::ArroySplit) = (x.k, )

# Base.Vector(w::Window) = collect(w.window)
# Base.convert(::Type{Vector}, w::Window) = Vector(w)
# Base.:(==)(w::Window, v::AbstractVector) = Vector(w) == v

Base.collect(w::Window) = SVector{w.nwin}(collect(w.window))
# Base.collect(w::Window) = collect(w.window)

get_window(wintype::Symbol, nwin::Int; kwargs...) = Window(wintype, nwin; kwargs...)

# mutable struct WinSetup{T<:AbstractFloat}
#     x::AbstractVector{T}
#     win::AbstractVector{T}
#     # frames::AbstractMatrix{T}
#     frame_starts::Vector{Int}
#     # wframes::AbstractMatrix{T}
#     nwin::Int

#     function WinSetup{T}(x::AbstractVector{T}, nwin::Int, nhops::Int) where T<:AbstractFloat
#         win = Vector{T}(undef, nwin)
#         # frames = Matrix{T}(undef, nwin, nhops)
#         frame_starts = collect(1:nwin-nhops:length(x)-nwin+1)
#         # wframes = Matrix{T}(undef, nwin, nhops)
#         new{T}(x, win, frame_starts, nwin)
#     end
# end

# function get_window(wintype::Symbol, nwin::Int; winperiod::Bool = true)
#     nwin == 1 && return [1.0]
#     winfunc = getproperty(WindowFunctions(), wintype)
#     winperiod && return (winfunc(x) for x in range(-0.5, stop=0.5, length=nwin+1)[1:end-1])
#     (winfunc(x) for x in range(-0.5, stop=0.5, length=nwin))
# end

# function get_window!(winsetup::WinSetup, wintype::Symbol; winperiod::Bool = true)
#     nwin = length(winsetup.win)
#     winsetup.win = @views collect(get_window(wintype, nwin; winperiod=winperiod))
# end

# function get_frames(x::AbstractVector{T}, win::AbstractVector{T}, nwin::Int, noverlap::Int) where T<:AbstractFloat
    
#     @views (x[start_idx:start_idx+nwin-1] .* win for start_idx in range(1, step=nhop, length=num_hops))
# end

# get_frames(ws::WinSetup) = collect(@view ws.x[ws.frame_starts[i]:ws.frame_starts[i]+ws.nwin-1] for i in 1:length(ws.frame_starts))
# get_wframes(ws::WinSetup) = (collect((@view ws.x[ws.frame_starts[i]:ws.frame_starts[i]+ws.nwin-1]) .* ws.nwin for i in 1:length(ws.frame_starts)))


# function get_wframes(x::AbstractVector{T}, nwin::Int, noverlap::Int) where T<:AbstractFloat
    
#     @views (x[start_idx:start_idx+nwin-1] for start_idx in range(1, step=nhop, length=num_hops))
# end

# function get_frames!(winsetup::WinSetup, nwin::Int, noverlap::Int)
#     nhop = nwin - noverlap
#     num_hops = div(length(winsetup.x) - nwin, nhop) + 1
    
#     winsetup.frames = collect(@views (winsetup.x[start_idx:start_idx+nwin-1] for start_idx in range(1, step=nhop, length=num_hops)))
# end

# function get_wframes!(winsetup::WinSetup, nwin::Int, noverlap::Int)

    
#     winsetup.wframes = collect(@views (winsetup.x[start_idx:start_idx+nwin-1] .* winsetup.win for start_idx in range(1, step=nhop, length=num_hops)))
# end

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
    num_hops = div(length(audio.data) - nwin, nhop) + 1
    # winsetup = WinSetup{eltype(audio.data)}(audio.data, nwin, num_hops)
    # winsetup.win = collect(get_window(wintype, nwin; winperiod=winperiod))
    # wframes = get_frames(audio.data, winsetup.win, nwin, noverlap)
    # get_window!(winsetup, wintype; winperiod=winperiod)
    # get_frames(winsetup)
    window = get_window(wintype, nwin; winperiod=winperiod)
    window
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

