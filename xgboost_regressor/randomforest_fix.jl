using Test
using MLJ, SoleXplorer
using DataFrames, Random
using MLJXGBoostInterface
using SoleModels
using XGBoost
const SX = SoleXplorer
const XGB = XGBoost
using DecisionTree

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xc, yc = @load_iris
Xc = DataFrame(Xc)

(train_ratio, seed, sampling_fraction, n_trees) = (0.5, 1, 0.7, 6)

model, mach, ds = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest, params=(;n_trees, sampling_fraction)),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio, rng=Xoshiro(seed)),
    measures=(accuracy, kappa, confusion_matrix),
)
sx_acc = model.measures.measures_values[1]
yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
mlj_acc = accuracy(yhat, ds.y[ds.tt[1].test])

sx_yhat = model.model[1].info.supporting_predictions
mlj_yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))

# index 8
classlabels  = (mach).fitresult[2][sortperm((mach).fitresult[3])]
featurenames = MLJ.report(mach).features
solem        = solemodel(MLJ.fitted_params(mach).forest; classlabels, featurenames)
apply!(solem, DataFrame(ds.X[ds.tt[1].test[8:10], :], ds.info.vnames), ds.y[ds.tt[1].test[8:10]])