using Revise, BenchmarkTools, Audio911
using SplitApplyCombine
using FFTW, DSP
using LinearAlgebra
using SpecialFunctions
using StatsBase
using Statistics
using Roots
using NaNStatistics
# using Polynomials
# using Plots

TESTPATH = joinpath(dirname(pathof(Audio911)), "..", "test")
TESTFILE = "common_voice_en_23616312.wav"
# TESTFILE = "104_1b1_Al_sc_Litt3200_4.wav"
wavfile = joinpath(TESTPATH, TESTFILE)

sr = 16000
audio = load_audio(source=wavfile, sr=sr, norm=true);

include("/home/paso/.julia/dev/Audio911.jl/src/utils/histogram.jl")

# ---------------------------------------------------------------------------------- #
#                                spectral functions                                  #
# ---------------------------------------------------------------------------------- #
sum_s = (s) -> sum(s, dims = 1)
sum_sfreq = (s, sfreq) -> sum(s .* sfreq, dims = 1)

_get_spec_centroid(s::AbstractArray{T}, sfreq::AbstractVector{T}) where {T <: AbstractFloat} = sum_sfreq(s, sfreq) ./ sum_s(s)

function _get_spec_spread(s::AbstractArray{T}, sfreq::AbstractVector{T}) where {T <: AbstractFloat}
	centroid = _get_spec_centroid(s, sfreq)
	vec(sqrt.(sum_s(s .* (sfreq .- centroid).^2) ./ sum_s(s)))
end

# ---------------------------------------------------------------------------------- #
#                                     utilities                                      #
# ---------------------------------------------------------------------------------- #
function moving_mean(x::AbstractVector{<:AbstractFloat}, w::Int64)
    isodd(w) || throw(ArgumentError("Window size must be odd."))
    n = length(x)
    pad = w ÷ 2
    map(i -> median(@view x[max(1, i - pad):min(n, i + pad)]), 1:n)
end

function f_peaks(n::Vector{T}) where {T <: AbstractFloat}
    n[end] = 0
    padded_n = [zeros(T, 3); n; zeros(T, 3)]
    nn = vcat(repeat([n; zeros(T, 7)], 2)..., n, zeros(T, 8), repeat([n; zeros(T, 7)], 2)..., n)
    temp = repeat(padded_n, outer=(1, 6))
    
    b = vec(all(reshape(nn, :, 6) .< temp, dims=2))
    findall(b) .- 3
end

function get_threshs_from_feature(feature::Vector{Float64}, bins::Int64, type::Symbol)
    hist_bins = max(10, round(Int, length(feature) / bins))
    m_feature = mean(feature)
    n_feature, edges_feature = get_histcounts(feature, nbins = hist_bins)

    if type == :specspread
        if edges_feature[1] == 0
            n_feature = @view n_feature[2:end]
            edges_feature = @view edges_feature[2:end]
        end
        minval = m_feature / 2
    else
        minval = minimum(feature)
    end

    peaks_idx = f_peaks(Float64.(n_feature))

    if type == :energy && length(peaks_idx) >= 2 && all(maximum(n_feature[peaks_idx[1:2]]) > m_feature)
        if edges_feature[peaks_idx[2]] > m_feature
            peaks_idx = peaks_idx[1:1]
        elseif edges_feature[peaks_idx[1]] > m_feature
            peaks_idx = @view peaks_idx[2:end]
        end
    end

    eF0 = [edges_feature; 0]
    AA = 0.5 * ([0; edges_feature] .- eF0) .+ eF0

    if isempty(peaks_idx)
        M1, M2 = m_feature / 2, minval
    elseif length(peaks_idx) == 1
        M1, M2 = AA[peaks_idx[1] + 1], minval
    else
        M1, M2 = AA[peaks_idx[2] + 1], AA[peaks_idx[1] + 1]
    end

    return M1, M2
end

function debuffer_frame_overlap(speech_mask::BitVector, nwin::Int64, noverlap::Int64)
    nhop = nwin - noverlap
    n_shared_frames = floor(Int, nwin / nhop)

    nearest_nv = DSP.filt(ones(n_shared_frames), 1, [speech_mask; zeros(n_shared_frames - 1)])

    n = length(nearest_nv)
    thresh = similar(nearest_nv)
    
    @views begin
        thresh[1:n_shared_frames-1] .= (1:n_shared_frames-1) ./ 2
        thresh[n_shared_frames:n-n_shared_frames+1] .= 1
        thresh[n-n_shared_frames+2:end] .= (n_shared_frames-1:-1:1) ./ 2
    end

    return nearest_nv .>= thresh, nhop
end

# ---------------------------------------------------------------------------------- #
#                                   speech detector                                  #
# ---------------------------------------------------------------------------------- #
function _speech_detector(;
        source::Audio,
        nfft::Int64 = 2 * round(Int, 0.03 * source.sr),
        nwin::Int64 = 0.5 * nfft,
        noverlap::Int64 = 0,
        wintype::Tuple{Symbol, Symbol} = (:hann, :periodic),
        norm::Symbol = :winmagnitude,
        thresholds::Tuple{Float64, Float64} = (-Inf, -Inf),
        merge_distance::Int64 = 5 * nwin
)
    noverlap < nwin || throwthrow(ArgumentError("Overlap length must be smaller than nwin: $nwin."))

    weight = 5
    bins = 15
    spread_threshold = 0.05
    lower_factor = 0.8
    filt_length = 5

    fspec = get_stft(source; nfft=nfft, nwin=nwin, noverlap=noverlap, wintype=wintype, norm=norm)
    # fspec = _get_stft(source.data, source.sr; nfft=nfft, nwin=nwin, noverlap=noverlap, wintype=wintype, norm=norm)

    energy = vec(abs2.(fspec.data.win)' * abs2.(combinedims(collect(fspec.data.frames))))
    filt_energy = moving_mean(moving_mean(energy, filt_length), filt_length)
    spread = _get_spec_spread(fspec.data.spec, fspec.data.freq) / (source.sr / 2)
    spread[energy .< spread_threshold] .= 0
    filt_spread = moving_mean(moving_mean(spread, filt_length), filt_length)

    if thresholds != (-Inf, -Inf)
        energy_thresh = thresholds[1]
        spread_thresh = thresholds[2]
    else
        e_m1, e_m2 = get_threshs_from_feature(filt_energy, bins, :energy)
        s_m1, s_m2 = get_threshs_from_feature(filt_spread, bins, :specspread)
        ww = 1 / (weight + 1)
        spread_thresh = ww * (weight * s_m2 + s_m1[1]) * lower_factor
        energy_Thresh = ww * (weight * e_m2 + e_m1[1])
    end

    speech_mask = @. (filt_spread > spread_thresh) & (filt_energy > energy_Thresh)
    noverlap > 0 && (speech_mask, nwin = debuffer_frame_overlap(speech_mask, nwin, noverlap))

    unbuff_out_mask = vcat(repeat(speech_mask', outer=(nwin, 1))[:], falses(length(source.data) - length(speech_mask) * nwin))
    difference = diff([unbuff_out_mask; false])

    idx_m1 = findall(==(-1), difference)
    idx_p1 = speech_mask[1] == 0 ? findall(==(1), difference) : vcat(1, findall(==(1), difference))

    testmask = length(idx_p1) > 1 ? idx_p1[2:end] .- idx_m1[1:(length(idx_p1) - 1)] .<= merge_distance : falses(0, 1)

    if isempty(idx_p1) || isempty(idx_m1)
        outidx = []
    else
        idx_p2 = idx_p1[2:end, :]
        idx_m2 = idx_m1[1:(length(idx_p1) - 1), :]
        amask = .!testmask
        outidx = reshape([idx_p1[1]; idx_p2[amask]; idx_m2[amask]; idx_m1[end]], :, 2)
    end

    Audio(reduce(vcat, [source.data[i[1]:i[2]] for i in eachrow(outidx)]), source.sr)
end

function speech_detector(source::Audio; kwargs...)
    _speech_detector(source=source, kwargs...)
end

# references
# https://github.com/linan2/Voice-activity-detection-VAD-paper-and-code

#----------------------------------------------------------------------------------#
#                                      DEBUG                                       #
#----------------------------------------------------------------------------------#
source = audio
wintype = (:hann, :periodic)
nwin = round(Int, 0.03 * source.sr)
noverlap = 0
norm = :winmagnitude
thresholds = (-Inf, -Inf)
merge_distance = nwin * 5
nfft = 2 * nwin
