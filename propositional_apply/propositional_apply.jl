using Test
using SoleXplorer, SoleModels
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
featurenames = MLJ.report(ds.mach).features
solem = SX.solemodel(MLJ.fitted_params(ds.mach).tree)

@btime apply!(solem,X,y)
# 202.379 μs (1073 allocations: 52.47 KiB)
@btime propositional_apply!(solem, X, y)
# 18.437 μs (504 allocations: 16.11 KiB)
# tipizzato
# 16.710 μs (500 allocations: 16.08 KiB)

apply!(solem.root,X,y)

# ## Test
# @btime SX.apply!(solem, X, y)
# # 105.046 μs (951 allocations: 48.75 KiB)
# # 67.695 μs (628 allocations: 29.69 KiB)

# #########################################################################
# ds = setup_dataset(
#     Xr, yr,
#     model=SX.RandomForestRegressor(n_trees=100),
#     resample=CV(nfolds=10, shuffle=true),
#     train_ratio=0.7,
#     rng=Xoshiro(1),
# )

# i = 1
# train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
# X_test, y_test = SX.get_X(ds)[test, :], SX.get_y(ds)[test]
# X, y = X_test, y_test

# MLJ.fit!(ds.mach, rows=train, verbosity=0)
# featurenames = MLJ.report(ds.mach).features
# solem = SX.solemodel(MLJ.fitted_params(ds.mach).forest)

# ## Test
# @btime SX.apply!(solem, X, y)
# # 203.951 ms (1969599 allocations: 89.97 MiB)
# # 200.996 ms (1967740 allocations: 89.65 MiB)


# ---------------------------------------------------------------------------- #
#                            apply!(solem, X, y)                               #
# ---------------------------------------------------------------------------- #
# julia> solem isa SoleModels.DecisionEnsemble
# false

dt = MLJ.fitted_params(ds.mach).raw_tree
# julia> typeof(dt.node)
# DecisionTree.Node{Float64, UInt32}
reference = apply_tree(dt.node, X)
# julia> typeof(MLJ.fitted_params(ds.mach).raw_tree)
# Root{Float64, UInt32}
AbstractModel = SX.AbstractModel

solem isa AbstractModel # true
X isa AbstractDataFrame    # true
y isa AbstractVector       # true

function propositional_apply!() end

using SoleModels: DecisionTree
function propositional_apply!(solem::SoleModels.DecisionTree{T}, X::AbstractDataFrame)::Nothing where T
    predictions = [propositional_apply!(solem.root, x) for x in eachrow(X)]
    set_predictions(solem, predictions)

    return nothing
end
# 361.024 ns (2 allocations: 656 bytes)

soler=solem.root
x = X[1, :]

get_featid(s::SoleModels.Branch{T}) where T = s.antecedent.value.metacond.feature.i_variable
get_cond(s::SoleModels.Branch{T}) where T = s.antecedent.value.metacond.test_operator
get_thr(s::SoleModels.Branch{T}) where T = s.antecedent.value.threshold
set_predictions(s,p) = solem.info.supporting_predictions = p

function propositional_apply!(soler::SoleModels.Branch{T}, x::DataFrameRow) where T
    featid, cond, thr = get_featid(soler), get_cond(soler), get_thr(soler)
    feature_value = x[featid]
    condition_result = cond(feature_value, thr)
    
    if condition_result
        return propositional_apply!(soler.posconsequent, x)  # or however left child is accessed
    else
        return propositional_apply!(soler.negconsequent, x)  # or however right child is accessed
    end
end

function propositional_apply!(leaf::SoleModels.ConstantModel{T}, x::DataFrameRow) where T
    leaf.outcome
end

# function al_apply!(solem::SoleModels.DecisionTree, X::AbstractDataFrame)
#     # predictions = Array{T}(undef, nrow(X))
#     predictions = Vector{Any}(undef, nrow(X))
#     for (i, x) in enumerate(eachrow(X))
#         # @show typeof(tree)
#         # predictions[i] = propositional_apply!(tree, x)
#         predictions[i] = x
#     end
#     return predictions
# end
# # 439.424 ns (17 allocations: 896 bytes)



