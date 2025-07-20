using Test
using SoleXplorer
using MLJ, DataFrames, Random
const SX = SoleXplorer

# using DecisionTree

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

modelr = symbolic_analysis(
    Xc, yc;
    model=RandomForestClassifier(n_trees=5),   
)

ds = setup_dataset(
    Matrix(Xr), yr,
    model=SX.RandomForestRegressor(n_trees=5,),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
)
SX.train_test(ds)
i = 1
train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
X_test, y_test = SX.get_X(ds)[test, :], SX.get_y(ds)[test]
X, y = X_test, y_test

MLJ.fit!(ds.mach, rows=train, verbosity=0)
classlabels  = string.(ds.mach.fitresult[2][sortperm((ds.mach).fitresult[3])])
featurenames = MLJ.report(ds.mach).features
solem        = SX.solemodel(MLJ.fitted_params(ds.mach).forest; classlabels, featurenames)
# 1.305 ms (8776 allocations: 938.11 KiB)

# #########################################################################
get_featid(s::SX.Branch{T}) where T = s.antecedent.value.metacond.feature.i_variable
get_cond(s::SX.Branch{T}) where T = s.antecedent.value.metacond.test_operator
get_thr(s::SX.Branch{T}) where T = s.antecedent.value.threshold

function set_predictions(
    info::NamedTuple,
    preds::Vector{T},
    y::AbstractVector{S}
)::NamedTuple where {T,S<: SX.Label}
    @show typeof(info)(merge(MLJ.params(info), (supporting_predictions=preds, supporting_labels=y)))
end

# #########################################################################
function propositional_apply!(
    solem::SX.DecisionTree{T},
    X::SX.AbstractDataFrame,
    y::AbstractVector{S}
)::Nothing where {T, S<:SX.CLabel}
    predictions = SX.CLabel[propositional_apply(solem.root, x) for x in eachrow(X)]
    solem.info = set_predictions(solem.info, predictions, y)
    return nothing
end

function propositional_apply!(
    solem::SX.DecisionTree{T},
    X::SX.AbstractDataFrame,
    y::AbstractVector{S}
)::Nothing where {T, S<:SX.RLabel}
    predictions = SX.RLabel[propositional_apply(solem.root, x) for x in @views eachrow(X)]
    solem.info = set_predictions(solem.info, predictions, y)
    return nothing
end

function propositional_apply(soler::SX.Branch{T}, x::DataFrameRow)::T where T
    featid, cond, thr = get_featid(soler), get_cond(soler), get_thr(soler)
    feature_value = x[featid]
    condition_result = cond(feature_value, thr)
    
    if condition_result
        return propositional_apply(soler.posconsequent, x)  # or however left child is accessed
    else
        return propositional_apply(soler.negconsequent, x)  # or however right child is accessed
    end
end

function propositional_apply(leaf::SX.ConstantModel{T}, ::DataFrameRow)::T where T
    leaf.outcome
end

# @btime propositional_apply!(solem, X, y)
# # 236.438 μs (7332 allocations: 222.55 KiB)

get_models(s::SX.DecisionEnsemble) = s.models
# #########################################################################
function propositional_apply!(
    solem::SX.DecisionEnsemble{T,S},
    X::AbstractDataFrame,
    y::AbstractVector;
    suppress_parity_warning=false
)::Nothing where {T,S}
    predictions = permutedims(hcat([propositional_apply(s, X, y) for s in get_models(solem)]...))
    predictions = [
        SX.weighted_aggregation(solem)(p; suppress_parity_warning)
        for p in eachcol(hcat(predictions...))
    ]
    solem.info = set_predictions(solem.info, predictions, y)
    return nothing
end

function propositional_apply(
    solebranch::SX.Branch{T},
    X::SX.AbstractDataFrame,
    y::AbstractVector{S}
)::Vector{SX.Label} where {T, S<:SX.Label}
    predictions = SX.Label[propositional_apply(solebranch, x) for x in eachrow(X)]
    solebranch.info = set_predictions(solebranch.info, predictions, y)
    return predictions
end

for s in get_models(solem)
    @show length(propositional_apply(s, X, y))
end



predictions = permutedims(hcat([propositional_apply(s, X, y) for s in get_models(solem)]...))
predictions = [
    SX.weighted_aggregation(solem)(p; suppress_parity_warning = false)
    for p in eachcol(predictions)
]
