using Revise, Audio911, BenchmarkTools, SIMD, FFTW, LinearAlgebra, DSP
using Base.Threads
# using DSP
# using Plots, Parameters, FFTW, DSP, StatsBase, NaNStatistics
# using Unitful, NamedArrays

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

audio = load_audio(source=wavfile, sr=16000)

x=audio.data
sr=16000
nfft=256
nwin=256
noverlap=128
wintype=(:hann, :symmetric)
norm=:power # :none, :power, :magnitude, :pow2mag

const NORM_FUNCS = Dict(
    :power => x -> (@. real((x * conj(x)))),
    :magnitude => x -> (@. abs(x)),
)

# ---------------------------------------------------------------------------- #
#                     windows - adapted from DSP package                       #
# ---------------------------------------------------------------------------- #
function get_window(winfunc::Function, nwin::Int; padding::Int = 0, zerophase::Bool = false, periodic::Bool = false)
    @assert nwin > 0 "nwin must be positive"
    @assert padding ≥ 0 "padding must be nonnegative"    

    if nwin == 1
        # if nwin is set to 1, no windowing will be applied. for future applications with wavelets.
        return 1.0
    elseif zerophase
        return vcat([winfunc.(range(0.0, stop=(nwin÷2)/nwin, length=nwin÷2+1))], [winfunc.(range(-(nwin÷2)/nwin, stop=-1/nwin, length=nwin÷2))])
    elseif periodic
        return vcat(winfunc.(range(-0.5, stop=0.5, length=nwin+1))[1:end-1])
    else
        return vcat(winfunc.(range(-0.5, stop=0.5, length=nwin)))
    end
end

function hann(nwin::Int; kwargs...)
    get_window(nwin; kwargs...) do x
        0.5 * (1 + cospi(2x))
    end
end

function buffer(x::AbstractVector{T}, nwin::Int, noverlap::Int) where T<:AbstractFloat
    nhop = nwin - noverlap
    num_hops = div(length(x) - nwin, nhop) + 1
    
    y = Matrix{T}(undef, nwin, num_hops)
    
    @threads for j in 1:num_hops
        start_idx = (j - 1) * nhop + 1
        @simd for i in 1:nwin
            @inbounds y[i, j] = x[start_idx + i - 1]
        end
    end

    return y
end

function _get_frames(x::AbstractVector{Float64}, window::Tuple{Symbol, Symbol}, nwin::Int64, noverlap::Int64)
    @assert window[2] in [:periodic, :symmetric] "window can be only :symmetric or :periodic"
	buffer(x, nwin, noverlap), getfield(Main, window[1])(nwin; periodic=window[2] == :periodic)
end

function _get_stft(;
    x::AbstractVector{T},
    sr::Int,
    nfft::Union{Int, AbstractFloat, Nothing} = nothing,
    nwin::Union{Int, AbstractFloat, Nothing} = nothing,
    noverlap::Union{Int, AbstractFloat, Nothing} = nothing,
    wintype::Tuple{Symbol, Symbol} = (:hann, :periodic),
    norm::Symbol = :power, # :none, :power, :magnitude
) where T<:AbstractFloat
    typeof(nfft) <: AbstractFloat && begin nfft = round(Int, nfft * sr) end
    typeof(nwin) <: AbstractFloat && begin nwin = round(Int, nwin * sr) end
    typeof(noverlap) <: AbstractFloat && begin noverlap = round(Int, noverlap * sr) end

    # apply default parameters if not provided
    nfft = nfft !== nothing ? nfft : sr !== nothing ? (sr <= 8000 ? 256 : 512) : 512
    nwin = nwin !== nothing ? nwin : nfft
    noverlap = noverlap !== nothing ? noverlap : round(Int, nwin * 0.5)

    @assert noverlap < nwin "Overlap length must be smaller than nwin: $nwin."
    @assert nwin ≤ nfft "FFT window size smaller than actual window size is highly discuraged."
    @assert haskey(NORM_FUNCS, norm) "Unknown spectrum_type: $norm."

    frames, win = _get_frames(x, wintype, nwin, noverlap)    
    wframes = nwin < nfft ? vcat(frames, zeros(Float64, nfft - nwin, size(frames, 2))) .* win : frames .* win
    # wnorm = sr * sum(abs2, win)

    nout = (nfft >> 1)+1
    out = zeros(ComplexF64, nout, size(wframes, 2))
    tmp = Vector{ComplexF64}(undef, nout)

    plan = plan_rfft(1:nfft)
    offset = 0

    for i in eachcol(wframes)
        mul!(tmp, plan, i)
        copyto!(out, offset+1, NORM_FUNCS[norm](tmp), 1, nout)
        offset += nout
    end
end


@btime DSP.stft(x, nwin, noverlap, nothing; onesided=true, nfft=nfft, fs=sr, window=hanning);
@btime get_stft(source=audio, sr=sr, nfft=nfft, nwin=nwin, noverlap=noverlap, wintype=wintype, norm=norm);
