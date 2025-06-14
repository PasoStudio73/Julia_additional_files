include("/home/paso/Documents/Aclai/Sole/SoleAudio.jl/src/SoleXplorer/SoleXplorer.jl")
using .SoleXplorer

using Catch22
using StatsBase: mean
using SoleData: aggregator_bottom, existential_aggregator

# using DataFrames, JLD2
# using Audio911
using ModalDecisionTrees
using SoleDecisionTreeInterface
using MLJ
# using MLJDecisionTreeInterface
# using Random

# using CategoricalArrays
# using Plots

# load jld2 debug
# jld2file = "/home/paso/Documents/Aclai/Sole/SoleAudio.jl/examples/dataframes/respiratory_Pneumonia.jld2"
# jld = jldopen(jld2file)
# X, y = jld["X"], jld["y"]
# df = MLD(X, y)
f1=[:mode_10, :min, :catch9, :centroid_freq, :periodicity]
f2=[:mode_10, :min, :catch9, :gino, :periodicity]
f3=[:custom]

# ---------------------------------------------------------------------------- #
#                           propositional structures                           #
# ---------------------------------------------------------------------------- #
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
    :std => std,
    (CATCH22_NAMES[i] => f for (i,f) in enumerate(catch22))...
)

const DEF_FEATS = Dict(
    :catch9 => Symbol[:max, :min, :mean, :med, :std, :stretch_high, :stretch_decreasing, :entropy_pairs, :transition_variance],
    :minmax => Symbol[:max, :min],
    :custom => Symbol[:max, :std, :mode_5, :acf_timescale, :ami2]
)

default = [:catch9]

abstract type AbstractPCond end
abstract type AbstractMCond end
abstract type AbstractMetaConds end

struct PCond <: AbstractPCond
    f::Base.Callable
end
Base.show(io::IO, c::PCond) = print(io, "$(c.f)")

struct MCond <: AbstractMCond
    op::Function
    f::Base.Callable
end
Base.show(io::IO, c::MCond) = print(io, "($(c.op), $(c.f))")
Base.Tuple(x::MCond) = (x.op, x.f)

struct MetaConds <: AbstractMetaConds
    f::Vector{Base.Callable}
    patched_f::Vector{Tuple{Function, Function}}
end

function collect_funcs(feats::AbstractVector{Symbol}, avail::Dict{Symbol, <:Base.Callable})#::Vector{Base.Callable}
    invalid_feats = filter(f -> f ∉ keys(avail), feats)
    isempty(invalid_feats) || throw(ArgumentError("Features not found: $(join(invalid_feats, ", "))"))

    pconds = PCond[]
    mconds = MCond[]
    for i in feats
        f = avail[i]
        push!(pconds, PCond(f))
        for (polarity, op) in ((:+, ≥), (:-, ≤))
            fname = Symbol(string(f, polarity))
            @eval function $fname(channel)
                val = $f(channel)
                isnan(val) ? aggregator_bottom(existential_aggregator($op), eltype(channel)) : eltype(channel)(val)
            end
            push!(mconds, MCond(op, get_patched_function(f, polarity)))
        end
    end
    pconds
    # [avail[k] for k in feats if k in keys(avail)]
    # return metaconditions
end

get_patched_function(f::Base.Callable, polarity::Symbol) = @eval $(Symbol(string(f)*string(polarity)))

# function get_patched_funcs(funcs::AbstractVector{<:Base.Callable})
#     patched_f = Tuple{Function, Function}[]

#     for f in funcs
#         for (polarity, op) in ((:+, ≥), (:-, ≤))
#             fname = Symbol(string(f, polarity))
#             @eval function $fname(channel)
#                 val = $f(channel)
#                 isnan(val) ? SoleData.aggregator_bottom(SoleData.existential_aggregator($op), eltype(channel)) : eltype(channel)(val)
#             end
#             push!(patched_f, (op, get_patched_function(f, polarity)))
#         end
#     end

#     return patched_f
# end

function get_funcs(feats::Union{AbstractVector{Symbol},Nothing})#::Tuple{Vector{Base.Callable}, Vector{Tuple{Function, Function}}}
    isnothing(feats) && begin
        println("No features specified. Using default features: $default")
        feats = default
    end
    feats = unique(vcat(filter(x -> x ∉ keys(DEF_FEATS), feats), DEF_FEATS[filter(x -> x in  keys(DEF_FEATS), feats)...]))
    # funcs = collect_funcs(feats, FEATS_DICT)
    # return MCondition(funcs, get_patched_funcs(funcs))

    collect_funcs(feats, FEATS_DICT)
end

a=get_funcs(f3)

# # ---------------------------------------------------------------------------- #
# #                              modal structures                                #
# # ---------------------------------------------------------------------------- #
# function mean_longstretch1(x) Catch22.SB_BinaryStats_mean_longstretch1((x)) end
# function diff_longstretch0(x) Catch22.SB_BinaryStats_diff_longstretch0((x)) end
# function quantile_hh(x) Catch22.SB_MotifThree_quantile_hh((x)) end
# function sumdiagcov(x) Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)) end

# function histogramMode_5(x) Catch22.DN_HistogramMode_5((x)) end
# function f1ecac(x) Catch22.CO_f1ecac((x)) end
# function histogram_even_2_5(x) Catch22.CO_HistogramAMI_even_2_5((x)) end

# function get_patched_feature(f::Base.Callable, polarity::Symbol)
#     if f in [minimum, maximum, StatsBase.mean, median]
#         f
#     else
#         @eval $(Symbol(string(f)*string(polarity)))
#     end
# end

# nan_guard = [:std, :mean_longstretch1, :diff_longstretch0, :quantile_hh, :sumdiagcov, :histogramMode_5, :f1ecac, :histogram_even_2_5]

# for f_name in nan_guard
#     @eval (function $(Symbol(string(f_name)*"+"))(channel)
#         val = $(f_name)(channel)

#         if isnan(val)
#             SoleData.aggregator_bottom(SoleData.existential_aggregator(≥), eltype(channel))
#         else
#             eltype(channel)(val)
#         end
#     end)
#     @eval (function $(Symbol(string(f_name)*"-"))(channel)
#         val = $(f_name)(channel)

#         if isnan(val)
#             SoleData.aggregator_bottom(SoleData.existential_aggregator(≤), eltype(channel))
#         else
#             eltype(channel)(val)
#         end
#     end)
# end

# d=[(≥, get_patched_function(maximum, :+)),            (≤, get_patched_function(maximum, :-)),]

# push!(d,(≥, get_patched_function(maximum, :+)),            (≤, get_patched_function(maximum, :-)),)

# modal_feature_dict = Dict(
#     :catch9 => [
#         (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
#         (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
#         (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
#         (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
#         (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
#         (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
#         (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
#         (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
#         (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
#     ],
#     :minmax => [
#         (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
#         (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
#     ],
#     :custom => [
#         (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
#         # (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
#         # (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
#         # (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
#         (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
#         # (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
#         # (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
#         # (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
#         # (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
#         (≥, get_patched_feature(histogramMode_5, :+)),    (≤, get_patched_feature(histogramMode_5, :-)),
#         (≥, get_patched_feature(f1ecac, :+)),             (≤, get_patched_feature(f1ecac, :-)),
#         (≥, get_patched_feature(histogram_even_2_5, :+)), (≤, get_patched_feature(histogram_even_2_5, :-)),
#     ]
# )

# # Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree

# # ---------------------------------------------------------------------------- #
# #                             propositional analysis                           #
# # ---------------------------------------------------------------------------- #
# function propositional_analisys(
#     df::MLD,
#     feats::Union{Dict, Nothing}=nothing,

#     train_ratio::Float64=0.8,
#     rng::AbstractRNG=Random.GLOBAL_RNG
# )
#     metaconditions = get_funcs(feats)

#     p_variable_names = [
#         string(m[1], f_dict_string[j], "(", m[2], ")", m[3])
#         for j in metaconditions
#         for m in [match(r_split, i) for i in variable_names]
#     ]
    
#     X_propos = DataFrame([name => Float64[] for name in [match(r_select, v)[1] for v in p_variable_names]])
#     push!(X_propos, vcat([vcat([map(func, Array(row)) for func in metaconditions]...) for row in eachrow(X)])...)

#     X_train, y_train, X_test, y_test = partitioning(X_propos, y; train_ratio=train_ratio, rng=rng)

#     @info("Propositional analysis: train model...")
#     learned_dt_tree = begin
#         model = Tree(; max_depth=-1, )
#         mach = machine(model, X_train, y_train) |> fit!
#         fitted_params(mach).tree
#     end
    
#     sole_dt = solemodel(learned_dt_tree)
#     apply!(sole_dt, X_test, y_test)
#     # printmodel(sole_dt; show_metrics = true, variable_names_map = variable_names)

#     sole_dt
# end