using SoleFeatures
using Test
using Sole
using Random
using CategoricalArrays, DataFrames

# Class = Union{CategoricalValue, String, Symbol, Real} # presente in SoleFeatures
# ---------------------------------------------------------------------------- #
#                             dataset preprocess                               #
# ---------------------------------------------------------------------------- #
# load a time-series dataset
X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# prepare dataset for feature selection
features = [mean, std]
nwindows = 5
X_features = @test_nowarn SoleFeatures.feature_selection_preprocess(X; features, nwindows)

# ---------------------------------------------------------------------------- #
#                           feature selection Fede                             #
# ---------------------------------------------------------------------------- #
# function feature_selection(
#     X::AbstractDataFrame,
#     y::Vector{<:Class};
#     fs_methods::AbstractVector{<:NamedTuple{(:selector, :limiter)}} = [
#         ( # STEP 1: unsupervised variance-based filter
#             selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
#             limiter = PercentageLimiter(0.5),
#         ),
#         ( # STEP 2: supervised Mutual Information filter
#             selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
#             limiter = PercentageLimiter(0.1),
#         ),
#         ( # STEP 3: group results by variable
#             selector = IdentityFilter(),
#             limiter = SoleFeatures.IdentityLimiter(),
#         ),
#     ],

y_coded  = @. CategoricalArrays.levelcode(y)

# ricreo fs_methods passato da funzione, l'idea è bella e va mantenuta
fs_methods = [
    ( # STEP 1: unsupervised variance-based filter
        selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
        limiter = PercentageLimiter(0.5),
    ),
    # ( # STEP 2: supervised Mutual Information filter
    #     selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
    #     limiter = PercentageLimiter(0.1),
    # ),
    # ( # STEP 3: group results by variable
    #     selector = IdentityFilter(),
    #     limiter = SoleFeatures.IdentityLimiter(),
    # ),
]
# meno bello
# tipo di aggregazione che si vuole alla fine
aggrby = (
    aggrby = tuple(1:length(split(first(names(X)), "@@@"))...),
    aggregatef = length, # NOTE: or mean, minimum, maximum to aggregate scores instead of just counting number of selected features for each group
    group_before_score = Val(true),
)

for (fsm, gfs_params) in zip(fs_methods, aggrby)
    @show fsm
    @show gfs_params
end

fs_mid_results = NamedTuple{(:score,:indices,:name2score,:group_aggr_func,:group_indices,:aggrby)}[]

# for (fsm, gfs_params) in zip(fs_methods, aggrby)
fsm = fs_methods[1]
gfs_params = aggrby

    # pick survived columns only
    current_dataset_col_slice = 1:DataFrames.ncol(X)
    for i in 1:length(fs_mid_results)
        current_dataset_col_slice = current_dataset_col_slice[fs_mid_results[i].indices]
    end
    currX = @view X[:,current_dataset_col_slice]

    dataset_param = isnothing(y) || is_unsupervised(fsm.selector) ? (currX,) : (currX, y)

    score, idxes, g_indices =
        if isnothing(gfs_params)
            # perform normal feature selection
            _fs(dataset_param..., fsm...)..., nothing
        else
            # perform aggregated feature selection
            sel_g_indices, g_indices, g_scores, grouped_variable_scores = _fsgroup(
                dataset_param..., fsm..., gfs_params.aggrby;
                groups_separator = groups_separator,
                aggregatef = gfs_params.aggregatef,
                group_before_score = gfs_params.group_before_score
            )

            # find indices to re-sort the scores of all variables to their
            #    original position in dataset columns
            old_sort = sortperm(vcat(g_indices...))
            vcat(vcat(grouped_variable_scores...)[old_sort]...), vcat(g_indices[sel_g_indices]...), g_indices
        end

    sort!(idxes)

    push!(fs_mid_results, (
        score = score,
        indices = idxes,
        name2score = Dict{String,Number}(names(currX) .=> score),
        group_aggr_func = isnothing(gfs_params) ? nothing : gfs_params.aggregatef,

        group_indices = g_indices,
        aggrby = isnothing(gfs_params) ? nothing : gfs_params.aggrby

    ))
# end


# ---------------------------------------------------------------------------- #
function _fsgroup(
    X::AbstractDataFrame,
    y::Union{AbstractVector,Nothing},
    selector::SoleFeatures.AbstractFeaturesSelector,
    limiter::SoleFeatures.AbstractLimiter,
    aggrby::Tuple{Vararg{Integer}};
    groups_separator::AbstractString = SoleFeatures.Experimental._SEPARATOR,
    aggregatef::Function = mean,
    group_before_score::Union{Val{true},Val{false}} = Val(true),
)::Tuple{Vector{Int},Vector{Vector{Int}},Vector{<:Real},Vector{Vector{<:Real}}}

selector, limiter = fsm

    g_indices = group_indices_by_column_names(X, aggrby; groups_separator = groups_separator)

    scores = []
    groups_score = Vector(undef, length(g_indices))
    if group_before_score isa Val{true}
        # === group and then evaluate score internally to each group ===
        for (i, cur_g_indices) in enumerate(g_indices)
            s = isnothing(y) || is_unsupervised(selector) ?
                SoleFeatures.score(X[:,cur_g_indices], selector) :
                SoleFeatures.score(X[:,cur_g_indices], y, selector)

            push!(scores, s) # save scores of variables of current group
            groups_score[i] = aggregatef(s) # save aggregated group score
        end
    else
        # === calculate scores for all variables and then group ===
        allscores = isnothing(y) || is_unsupervised(selector) ?
            SoleFeatures.score(X, selector) :
            SoleFeatures.score(X, y, selector)

        for (i, cur_g_indices) in enumerate(g_indices)
            push!(scores, allscores[cur_g_indices]) # save scores of variables of current group
            groups_score[i] = aggregatef(allscores[cur_g_indices]) # save aggregated group score
        end
    end

    # convert groups_score from Vector{Any} type to Vector{Type of first element} type
    groups_score = convert.(typeof(groups_score[1]), groups_score)

    # # apply limiter on groups
    sel_idxes = SoleFeatures.limit(groups_score, limiter)

    # first element: index of the selected groups
    # second element: indices of variables for each group
    # third element: score of each group
    # fourth element: score of each variable grouped
    return sel_idxes, g_indices, groups_score, scores
end
function _fsgroup(
    X::AbstractDataFrame,
    selector::SoleFeatures.AbstractFeaturesSelector,
    limiter::SoleFeatures.AbstractLimiter,
    aggrby::Tuple{Vararg{Integer}};
    kwargs...
)::Tuple{Vector{Int},Vector{Vector{Int}},Vector{<:Real},Vector{Vector{<:Real}}}
    return _fsgroup(X, nothing, selector, limiter, aggrby; kwargs...)
end

########################

col_feats = unique([first(col).feats for col in eachcol(X_features)])

function filter_by_feature_type(X::DataFrame, feature_type::Symbol)
    X[:, [all(x -> x.feats == feature_type, col) for col in eachcol(X)]]
end

