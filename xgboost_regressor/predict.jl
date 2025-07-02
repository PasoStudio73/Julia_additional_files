using Test
using MLJ, SoleXplorer
using DataFrames, Random
# using SoleData, SoleModels
using XGBoost
import MLJBase: predict, predict_mean, predict_mode, predict_median, predict_joint
const SX = SoleXplorer
const XGB = XGBoost
const AbstractModel = SX.AbstractModel


Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#                                  from models                                 #
# ---------------------------------------------------------------------------- #
# decisiontree
learn_method = (
    (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).tree); apply!(solem, X, y); solem),
    (mach, X, y) -> (solem = solemodel(MLJ.fitted_params(mach).best_fitted_params.tree); apply!(solem, X, y); solem)
)
# randomforest
learn_method = (
    (mach, X, y) -> begin
        classlabels  = (mach).fitresult[2][sortperm((mach).fitresult[3])]
        featurenames = MLJ.report(mach).features
        solem        = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
        apply!(solem, X, y)
        return solem
    end,
    (mach, X, y) -> begin
        classlabels  = (mach).fitresult.fitresult[2][sortperm((mach).fitresult.fitresult[3])]
        featurenames = MLJ.report(mach).best_report.features
        solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.forest; classlabels, featurenames)
        apply!(solem, X, y)
        return solem
    end
)
# adaboost
learn_method = (
    (mach, X, y) -> begin
        weights      = mach.fitresult[2]
        classlabels  = sort(mach.fitresult[3])
        featurenames = MLJ.report(mach).features
        solem        = solemodel(MLJ.fitted_params(mach).stumps; weights, classlabels, featurenames)
        apply!(solem, X, y)
        return solem
    end,
    (mach, X, y) -> begin
        weights      = mach.fitresult.fitresult[2]
        classlabels  = sort(mach.fitresult.fitresult[3])
        featurenames = MLJ.report(mach).best_report.features
        solem        = solemodel(MLJ.fitted_params(mach).best_fitted_params.stumps; weights, classlabels, featurenames)
        apply!(solem, X, y)
        return solem
    end
)
# modaldecisiontree
learn_method = (
    (mach, X, y) -> ((_, solem) = MLJ.report(mach).sprinkle(X, y); solem),
    (mach, X, y) -> ((_, solem) = MLJ.report(mach).best_report.sprinkle(X, y); solem)
)
# modalrandomforest
learn_method = (
    (mach, X, y) -> ((_, solem) = MLJ.report(mach).sprinkle(X, y); solem),
    (mach, X, y) -> ((_, solem) = MLJ.report(mach).best_report.sprinkle(X, y); solem)
)
# xgboost
learn_method = (
    (mach, X, y) -> begin
        trees        = XGB.trees(mach.fitresult[1])
        featurenames = mach.report.vals[1].features
        solem        = solemodel(trees, @views(Matrix(X)), @views(y); featurenames)
        apply!(solem, mapcols(col -> Float32.(col), X), @views(y))
        return solem
    end,
    (mach, X, y) -> begin
        trees        = XGB.trees(mach.fitresult.fitresult[1])
        featurenames = mach.fitresult.report.vals[1].features
        solem        = solemodel(trees, @views(Matrix(X)), @views(y); featurenames)
        apply!(solem, mapcols(col -> Float32.(col), X), @views(y))
        return solem
    end
)

# ---------------------------------------------------------------------------- #
#                                    testing                                   #
# ---------------------------------------------------------------------------- #
model = symbolic_analysis(Xc, yc; preprocess=(;rng=Xoshiro(1)), measures=(log_loss, accuracy, confusion_matrix, kappa))

model, mach, ds = train_test(Xc, yc; preprocess=(;rng=Xoshiro(1)))
model, mach, ds = train_test(Xr, yr; preprocess=(;rng=Xoshiro(1)))
y = @views(ds.y)
tt = ds.tt

using MLJBase
yhat = SX.get_yhat(model)
measures   = MLJBase._actual_measures([SX.get_setup_meas(model)...], SX.get_solemodel(model))
operations = SX.get_operations(measures, SX.get_prediction_type(model))

nfolds = length(yhat)
test_fold_sizes = [length(yhat[k]) for k in 1:nfolds]
nmeasures = length(SX.get_setup_meas(model))

# measurements_vector = mapreduce(vcat, 1:nfolds) do k
#     yhat_given_operation = Dict(op=>op(mach, yhat[k]) for op in unique(operations))
#     # yhat_given_operation = Dict(op=>op(get_solemodel(model), mach, rows=yhat[k].test) for op in unique(operations))
#     test = yhat[k].test

#     [map(measures, operations) do m, op
#         m(
#             yhat_given_operation[op],
#             y[test],
#             # MLJBase._view(weights, test),
#             # class_weights
#             MLJBase._view(nothing, test),
#             nothing
#         )
#     end]
# end

# vedi in mlj l'output di measurement vector, secondo me gia l'abbiamo
Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree = Tree()
mljdt = evaluate(
    tree, Xr, yr;
    resampling=CV(shuffle=false),
    measures=[log_loss, accuracy, confusion_matrix],
    per_observation=true,
    verbosity=0
)

yhat = predict(mach, Xc[ds.tt[1].test, :])
sxt = SX.sole_predict(mach, model.model[1])
a = log_loss(yhat, yc[ds.tt[1].test])
b = log_loss(sxt, yc[ds.tt[1].test])