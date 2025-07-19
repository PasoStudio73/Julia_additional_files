using Test
using SoleXplorer
using MLJ, DataFrames, Random
const SX = SoleXplorer

using DecisionTree

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

ds = setup_dataset(
    Xc, yc,
    model=SX.DecisionTreeClassifier(),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
)

i = 1
train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
X_test, y_test = SX.get_X(ds)[test, :], SX.get_y(ds)[test]
X, y = X_test, y_test

MLJ.fit!(ds.mach, rows=train, verbosity=0)
solem = SX.solemodel(MLJ.fitted_params(ds.mach).tree)

# #########################################################################
get_featid(s::SX.Branch{T}) where T = s.antecedent.value.metacond.feature.i_variable
get_cond(s::SX.Branch{T}) where T = s.antecedent.value.metacond.test_operator
get_thr(s::SX.Branch{T}) where T = s.antecedent.value.threshold

function set_predictions(
    info::NamedTuple,
    preds::Vector{T},
    y::AbstractVector{S}
)::NamedTuple where {T,S<: SX.CLabel}
    typeof(info)(merge(MLJ.params(info), (supporting_predictions=preds, supporting_labels=y)))
end

function propositional_apply!(
    solem::SX.DecisionTree{T},
    X::SX.AbstractDataFrame,
    y::AbstractVector{S}
)::Nothing where {T, S<:SX.CLabel}
    predictions = String[propositional_apply!(solem.root, x) for x in eachrow(X)]
    solem.info = set_predictions(solem.info, predictions, y)
    return nothing
end

function propositional_apply!(soler::SX.Branch{T}, x::DataFrameRow)::T where T
    featid, cond, thr = get_featid(soler), get_cond(soler), get_thr(soler)
    feature_value = x[featid]
    condition_result = cond(feature_value, thr)
    
    if condition_result
        return propositional_apply!(soler.posconsequent, x)  # or however left child is accessed
    else
        return propositional_apply!(soler.negconsequent, x)  # or however right child is accessed
    end
end

function propositional_apply!(leaf::SX.ConstantModel{T}, ::DataFrameRow)::T where T
    leaf.outcome
end




