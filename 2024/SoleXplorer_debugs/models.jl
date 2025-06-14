using MLJDecisionTreeInterface
using ModalDecisionTrees
using SoleData
using Random

# ---------------------------------------------------------------------------- #
#                              available models                                #
# ---------------------------------------------------------------------------- #
const AVAIL_MODELS = Dict(
    :decision_tree => (
        method = MLJDecisionTreeInterface.DecisionTreeClassifier,

        params = (;
            max_depth=-1, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            min_purity_increase=0.0, 
            n_subfeatures=0, 
            post_prune=false, 
            merge_purity_threshold=1.0, 
            display_depth=5, 
            feature_importance=:impurity, 
            rng=Random.TaskLocalRNG()
        ),
        datatype = :aggregate
    ),
    :win_decision_tree => (
        method = MLJDecisionTreeInterface.DecisionTreeClassifier,

        params = (;
            max_depth=-1, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            min_purity_increase=0.0, 
            n_subfeatures=0, 
            post_prune=false, 
            merge_purity_threshold=1.0, 
            display_depth=5, 
            feature_importance=:impurity, 
            rng=Random.TaskLocalRNG()
        ),
        datatype = :reduce_aggregate
    ),
    :modal_decision_tree => (
        method = ModalDecisionTree,

        params = (;
            max_depth=nothing, 
            min_samples_leaf=4, 
            min_purity_increase=0.002, 
            max_purity_at_leaf=Inf, 
            max_modal_depth=nothing, 
            relations=nothing, 
            features=nothing, 
            conditions=nothing, 
            featvaltype=Float64, 
            initconditions=nothing, 
            # downsize=SoleData.var"#downsize#482"(), 
            print_progress=false, 
            display_depth=nothing, 
            min_samples_split=nothing, 
            n_subfeatures=identity, 
            post_prune=false, 
            merge_purity_threshold=nothing, 
            feature_importance=:split,
            rng=Random.TaskLocalRNG()
        ),
        datatype = :reducesize
    )
)

# ---------------------------------------------------------------------------- #
#                                   get model                                  #
# ---------------------------------------------------------------------------- #
function get_model(model_name::Symbol; kwargs...)
    !haskey(AVAIL_MODELS, model_name) && throw(ArgumentError("Model $model_name not found in available models. Valid options are: $(keys(AVAIL_MODELS))"))

    params = AVAIL_MODELS[model_name].params
    valid_kwargs = filter(kv -> kv.first in keys(params), kwargs)
    
    merge(params, valid_kwargs)
    AVAIL_MODELS[model_name].method(; merge(params, valid_kwargs)...)
end

model_name = :modal_decision_tree
a=get_model(model_name)