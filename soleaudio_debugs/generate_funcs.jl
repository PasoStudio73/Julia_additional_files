using SoleData: aggregator_bottom, existential_aggregator
using StatsBase
using Catch22

# ---------------------------------------------------------------------------- #
#                        catch22 pretty named functions                        #
# ---------------------------------------------------------------------------- #
mode_5(x) = Catch22.DN_HistogramMode_5((x))
mode_10(x) = Catch22.DN_HistogramMode_10((x))
embedding_dist(x) = Catch22.CO_Embed2_Dist_tau_d_expfit_meandiff((x))
acf_timescale(x) = Catch22.CO_f1ecac((x))
acf_first_min(x) = Catch22.CO_FirstMin_ac((x))
ami2(x) = Catch22.CO_HistogramAMI_even_2_5((x))
trev(x) = Catch22.CO_trev_1_num((x))
outlier_timing_pos(x) = Catch22.DN_OutlierInclude_p_001_mdrmd((x))
outlier_timing_neg(x) = Catch22.DN_OutlierInclude_n_001_mdrmd((x))
whiten_timescale(x) = Catch22.FC_LocalSimple_mean1_tauresrat((x))
forecast_error(x) = Catch22.FC_LocalSimple_mean3_stderr((x))
ami_timescale(x) = Catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi((x))
high_fluctuation(x) = Catch22.MD_hrv_classic_pnn40((x))
stretch_decreasing(x) = Catch22.SB_BinaryStats_diff_longstretch0((x))
stretch_high(x) = Catch22.SB_BinaryStats_mean_longstretch1((x))
entropy_pairs(x) = Catch22.SB_MotifThree_quantile_hh((x))
rs_range(x) = Catch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1((x))
dfa(x) = Catch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1((x))
low_freq_power(x) = Catch22.SP_Summaries_welch_rect_area_5_1((x))
centroid_freq(x) = Catch22.SP_Summaries_welch_rect_centroid((x))
transition_variance(x) = Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x))
periodicity(x) = Catch22.PD_PeriodicityWang_th0_01((x))

features = [minimum, maximum, StatsBase.cov, centroid_freq]

macro generate_patch(fname, polarity, op)
    return quote
        function $(Symbol(string(fname, polarity)))(channel)
            val = $(esc(fname))(channel)
            isnan(val) ? aggregator_bottom(existential_aggregator($(esc(op))), eltype(channel)) : eltype(channel)(val)
        end
    end
end

macro generate_feature_pairs(features)
    return quote
        [
            (op, $(esc(Symbol(String(Symbol(f) * polarity)))))
            for f in $(esc(features))
            for (op, polarity) in [(≥, :+), (<, :-)]
        ]
    end
end


for f in features
    for (polarity, op) in ((:A, ≥), (:B, <))
        # @eval @generate_patch f polarity op
        @eval function $(Symbol(string(f)*string(polarity)))(channel)
            val = $(f)(channel)
            isnan(val) ? aggregator_bottom(existential_aggregator($op), eltype(channel)) : eltype(channel)(val)
        end
    end
end

minimumA([1,2,3])

# function patch_feats(features::AbstractVector{Function})
    # for f in features
    #     for (polarity, op) in ((:A, ≥), (:B, <))
    #         @eval @generate_patch f polarity op

    #     minimumA([1,2,3])
    #     end
    # end
# end

macro generate_patch(fname, polarity, op)
    fname_str = String(Symbol(fname))
    new_fname = Symbol(fname_str * string(polarity))
    
    return quote
        function $(esc(new_fname))(channel)
            val = $(esc(fname))(channel)
            isnan(val) ? aggregator_bottom(existential_aggregator($(esc(op))), eltype(channel)) : eltype(channel)(val)
        end
    end
end

features = [std, maximum, StatsBase.cov, centroid_freq]

for f in features
    for (polarity, op) in ((:A, ≥), (:B, <))
        @generate_patch f polarity op
    end
end

# macro generate_patches(features)
#     expr = quote end
    
#     for f in eval(features)
#         for (polarity, op) in ((:+, ≥), (:-, <))
#             push!(expr.args, quote
#                 function $(esc(Symbol(String(Symbol(f)) * String(polarity))))(channel)
#                     val = $f(channel)
#                     isnan(val) ? aggregator_bottom(existential_aggregator($(esc(op))), eltype(channel)) : eltype(channel)(val)
#                 end
#             end)
#         end
#     end
    
#     return expr
# end

# macro generate_patches(features)
#     Expr(:block, [
#         quote
#             function $(esc(Symbol(String(Symbol(f)) * polarity)))(channel)
#                 val = $f(channel)
#                 isnan(val) ? aggregator_bottom(
#                     existential_aggregator($op), 
#                     eltype(channel)
#                 ) : eltype(channel)(val)
#             end
#         end
#         for f in @eval features
#         for (polarity, op) in (("+", ≥), ("-", <))
#     ]...)
# end




features = [minimum, maximum, StatsBase.cov, centroid_freq]
@generate_patches features


function model_closure(model, symbol)
    (args...; kwargs...) -> getproperty(model, symbol)(model, args...; kwargs...)
end