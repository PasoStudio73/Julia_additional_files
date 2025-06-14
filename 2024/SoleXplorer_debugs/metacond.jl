# ---------------------------------------------------------------------------- #
#                                features dicts                                #
# ---------------------------------------------------------------------------- #
using Catch22

const CATCH22_NAMES = Symbol[
    :mode_5, :mode_10, :embedding_dist, :acf_timescale, :acf_first_min, :ami2, :trev,
    :outlier_timing_pos, :outlier_timing_neg, :whiten_timescale, :forecast_error,
    :ami_timescale, :high_fluctuation, :stretch_decreasing, :stretch_high, :entropy_pairs,
    :rs_range, :dfa, :low_freq_power, :centroid_freq, :transition_variance, :periodicity
]

const FEATS_DICT = Dict(
    :max => maximum,
    :min => minimum,
    :mean => mean,
    :med => median,
    :var => var,
    :std => std,
    (CATCH22_NAMES[i] => f for (i,f) in enumerate(catch22))...
)

const DEF_FEATS = Dict(
    :catch9 => Symbol[:max, :min, :mean, :med, :std, :stretch_high, :stretch_decreasing, :entropy_pairs, :transition_variance],
    :minmax => Symbol[:max, :min],
    :custom => Symbol[:max, :std, :mode_5, :acf_timescale, :ami2]
)

default = [:catch9]

function collect_funcs(feats::AbstractVector{Symbol}, avail::Dict{Symbol, <:Base.Callable})::Vector{Function}
    invalid_feats = filter(f -> f ∉ keys(avail), feats)
    isempty(invalid_feats) || throw(ArgumentError("Features not found: $(join(invalid_feats, ", "))"))
    [avail[k] for k in feats if k in keys(avail)]
end

get_patched_function(f::Base.Callable, polarity::Symbol) = @eval $(Symbol(string(f)*string(polarity)))

function get_patched_funcs(funcs::AbstractVector{<:Base.Callable})::Vector{Tuple{Function, Function}}
    patched_f = Tuple{Function, Function}[]

    for f in funcs
        for (polarity, op) in ((:+, ≥), (:-, ≤))
            fname = Symbol(string(f, polarity))
            @eval function $fname(channel)
                val = $f(channel)
                isnan(val) ? SoleData.aggregator_bottom(SoleData.existential_aggregator(op), eltype(channel)) : eltype(channel)(val)
            end
            push!(patched_f, (op, get_patched_function(f, polarity)))
        end
    end

    return patched_f
end

function get_funcs(feats::Union{AbstractVector{Symbol},Nothing})::Tuple{Vector{Function}, Vector{Tuple{Function, Function}}}
    isnothing(feats) && begin
        println("No features specified. Using default features: $default")
        feats = default
    end
    def_f = filter(x -> x in  keys(DEF_FEATS), feats)
    if !isempty(def_f)
        def_f = DEF_FEATS[def_f...]
    end
    feats = unique(vcat(filter(x -> x ∉ keys(DEF_FEATS), feats), def_f))
    funcs = collect_funcs(feats, FEATS_DICT)
    return funcs, get_patched_funcs(funcs)
end