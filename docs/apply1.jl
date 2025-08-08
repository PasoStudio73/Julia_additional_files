# ---------------------------------------------------------------------------- #
#                                 File summary                                 #
# ---------------------------------------------------------------------------- #
# apply.jl — Unified conversion and application layer
#
# Provides `apply(ds, X, y)` methods that:
# - transform fitted MLJ machines from supported model packages into SoleModels
#   symbolic models (via `solemodel`)
# - build a logical dataset (via `scalarlogiset`)
# - apply the symbolic model to the dataset (via `apply!`)
#
# Supported model families:
# - DecisionTree.jl: DecisionTree{Classifier,Regressor}, RandomForest{Classifier,Regressor},
#   AdaBoostStumpClassifier
# - ModalDecisionTrees.jl: ModalDecisionTree, ModalRandomForest, ModalAdaBoost
# - XGBoost.jl: XGBoost{Classifier,Regressor}
#
# Common signature:
#     apply(ds, X::AbstractDataFrame, y::AbstractVector) -> symbolic model
#
# Arguments:
# - ds: PropositionalDataSet or ModalDataSet wrapping an MLJ.Machine; tuned variants
#       use MLJTuning.EitherTunedModel and access best_* fields in report/params.
# - X: feature table (DataFrame-compatible).
# - y: target vector used when building/applying the symbolic model.
#
# Notes:
# - Classification paths preserve class ordering/names via `classlabels`.
# - For XGBoost regressors, `base_score` is propagated so scores/predictions match
#   the trained model.
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
"""
Alias for MLJ machines used throughout this file.
"""
const Mach  = MLJ.Machine

"""
Type function returning the dataset wrapper union for tuned MLJ models of `T`.

Returns:
- Union{PropositionalDataSet{<:EitherTunedModel{<:Any,<:T}},
        ModalDataSet{<:EitherTunedModel{<:Any,<:T}}}
"""
TunedMach(T) = Union{
    PropositionalDataSet{<:MLJTuning.EitherTunedModel{<:Any, <:T}},
    ModalDataSet{<:MLJTuning.EitherTunedModel{<:Any, <:T}},
}

"""
Datasets handled by the DecisionTree single-tree path
(DecisionTreeClassifier, DecisionTreeRegressor).
"""
const DecisionTreeApply = Union{
    PropositionalDataSet{DecisionTreeClassifier}, 
    PropositionalDataSet{DecisionTreeRegressor},
}

"""
Datasets handled by the tuned DecisionTree single-tree path.
"""
const TunedDecisionTreeApply = Union{
    TunedMach(DecisionTreeClassifier),
    TunedMach(DecisionTreeRegressor)
}

"""
Datasets handled by the ModalDecisionTrees family (modal tree/forest/boosting).
"""
const ModalDecisionTreeApply = Union{
    ModalDataSet{ModalDecisionTree},
    ModalDataSet{ModalRandomForest},
    ModalDataSet{ModalAdaBoost}
}

"""
Datasets handled by tuned ModalDecisionTrees models.
"""
const TunedModalDecisionTreeApply = Union{
    TunedMach(ModalDecisionTree),
    TunedMach(ModalRandomForest),
    TunedMach(ModalAdaBoost)
}

# ---------------------------------------------------------------------------- #
#                              xgboost utilities                               #
# ---------------------------------------------------------------------------- #
"""
get_base_score(m::MLJ.Machine) -> Union{Number,Nothing}

Extract `base_score` from an MLJ machine, transparently handling tuned models.
Returns `nothing` if the underlying model does not define `base_score`.
"""
function get_base_score(m::MLJ.Machine)
    if m.model isa MLJTuning.EitherTunedModel
        return hasproperty(m.model.model, :base_score) ? m.model.model.base_score : nothing
    else
        return hasproperty(m.model, :base_score) ? m.model.base_score : nothing
    end
end

"""
Build a mapping from internal class indices to class labels as seen by MLJ.
Used to recover ordered `classlabels` for classification models.
"""
get_encoding(classes_seen) = Dict(MLJ.int(c) => c for c in MLJ.classes(classes_seen))

"""
Return ordered string class labels from an encoding dictionary.
The order matches the model’s internal class index order.
"""
get_classlabels(encoding)  = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

# ---------------------------------------------------------------------------- #
#                             DecisionTree package                             #
# ---------------------------------------------------------------------------- #
"""
Apply a single DecisionTree (classifier or regressor).

- Uses `MLJ.report(ds.mach).features` as `featurenames`.
- Builds a symbolic model from `fitted_params(...).tree`.
- Applies the model to a propositional logical dataset.
"""
function apply(
    ds :: DecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    solem        = solemodel(MLJ.fitted_params(ds.mach).tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply a tuned DecisionTree (classifier or regressor), using the best model/report.
"""
function apply(
    ds :: TunedDecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    solem = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

# randomforest
"""
Apply a RandomForestClassifier.

- Recovers ordered `classlabels` from `fitresult` to preserve class index mapping.
- Builds a forest symbolic model and applies it to a propositional logical dataset.
"""
function apply(
    ds :: PropositionalDataSet{RandomForestClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    classlabels  = ds.mach.fitresult[2][sortperm((ds.mach).fitresult[3])]
    featurenames = MLJ.report(ds.mach).features
    solem        = solemodel(MLJ.fitted_params(ds.mach).forest; classlabels, featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply a tuned RandomForestClassifier (best model/report).
"""
function apply(
    ds :: TunedMach(RandomForestClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    classlabels  = ds.mach.fitresult.fitresult[2][sortperm((ds.mach).fitresult.fitresult[3])]
    featurenames = MLJ.report(ds.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.forest; classlabels, featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply a RandomForestRegressor.
"""
function apply(
    ds :: PropositionalDataSet{RandomForestRegressor},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    solem        = solemodel(MLJ.fitted_params(ds.mach).forest; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply a tuned RandomForestRegressor (best model/report).
"""
function apply(
    ds :: TunedMach(RandomForestRegressor),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.forest; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

# adaboost
"""
Apply an AdaBoostStumpClassifier.

- Extracts stump `weights` and ordered `classlabels`.
- Builds a symbolic ensemble from `fitted_params(...).stumps`.
"""
function apply(
    ds :: PropositionalDataSet{AdaBoostStumpClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    weights      = ds.mach.fitresult[2]
    classlabels  = sort(string.(ds.mach.fitresult[3]))
    featurenames = MLJ.report(ds.mach).features
    solem        = solemodel(MLJ.fitted_params(ds.mach).stumps; weights, classlabels, featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply a tuned AdaBoostStumpClassifier (best model/report).
"""
function apply(
    ds :: TunedMach(AdaBoostStumpClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    weights      = ds.mach.fitresult.fitresult[2]
    classlabels  = sort(ds.mach.fitresult.fitresult[3])
    featurenames = MLJ.report(ds.mach).best_report.features
    solem        = solemodel(MLJ.fitted_params(ds.mach).best_fitted_params.stumps; weights, classlabels, featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                           ModalDecisionTrees package                         #
# ---------------------------------------------------------------------------- #
"""
Apply modal tree/forest/boosting models.

Delegates to the model’s `sprinkle(X, y)` report utility and returns the
constructed symbolic model.
"""
function apply(
    ds :: ModalDecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    (_, solem) = MLJ.report(ds.mach).sprinkle(X, y)
    return solem
end

"""
Apply tuned modal models using the best tuned report.
"""
function apply(
    ds :: TunedModalDecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    (_, solem) = MLJ.report(ds.mach).best_report.sprinkle(X, y)
    return solem
end

# ---------------------------------------------------------------------------- #
#                                XGBoost package                               #
# ---------------------------------------------------------------------------- #
"""
Apply an XGBoostClassifier.

- Extracts trees and MLJ class encoding to produce ordered `classlabels`.
- Uses `report.vals[1].features` as `featurenames`.
- Converts X to Float32 when building the logical dataset.
"""
function apply(
    ds :: PropositionalDataSet{XGBoostClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    trees        = XGBoost.trees(ds.mach.fitresult[1])
    encoding     = get_encoding(ds.mach.fitresult[2])
    classlabels  = get_classlabels(encoding)
    featurenames = ds.mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply a tuned XGBoostClassifier (best model/report).
"""
function apply(
    ds :: TunedMach(XGBoostClassifier),
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    trees        = XGBoost.trees(ds.mach.fitresult.fitresult[1])
    encoding     = get_encoding(ds.mach.fitresult.fitresult[2])
    classlabels  = get_classlabels(encoding)
    featurenames = ds.mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; classlabels, featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y)
    return solem
end

"""
Apply an XGBoostRegressor.

- Propagates/infer `base_score` when needed to reproduce model behavior.
- Builds a symbolic ensemble from extracted trees and applies it.
"""
function apply(
    ds :: PropositionalDataSet{XGBoostRegressor},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    base_score = get_base_score(ds.mach) == -Inf ? mean(ds.y[train]) : 0.5
    ds.mach.model.base_score = base_score

    trees        = XGBoost.trees(ds.mach.fitresult[1])
    featurenames = ds.mach.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y; base_score)
    return solem
end

"""
Apply a tuned XGBoostRegressor (best model/report), handling `base_score`.
"""
function apply(
    ds :: TunedMach(XGBoostRegressor),
    X  :: AbstractDataFrame,
    y  :: AbstractVector,
)
    base_score = get_base_score(ds.mach) == -Inf ? mean(ds.y[train]) : 0.5
    ds.mach.model.model.base_score = base_score

    trees        = XGBoost.trees(ds.mach.fitresult.fitresult[1])
    featurenames = ds.mach.fitresult.report.vals[1].features
    solem        = solemodel(trees, Matrix(X), y; featurenames)
    logiset      = scalarlogiset(mapcols(col -> Float32.(col), X), allow_propositional = true)
    apply!(solem, logiset, y; base_score)
    return solem
end