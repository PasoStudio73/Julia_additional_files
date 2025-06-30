using Test
using DataFrames, Random

using SoleModels
using SoleLogics
using SoleData
using SoleBase
using MultiData
using SoleXplorer
const SX = SoleXplorer

using MLJ
using MLJXGBoostInterface
using DecisionTree
const DT = DecisionTree
using XGBoost
const XGB = XGBoost

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                    matrixtable based logiset: benchmarks                     #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                            classification models                             #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:randomforest, params=(;n_trees=50)),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:adaboost),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)

model, _, _ = symbolic_analysis(
    Xc, yc;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(log_loss, accuracy, kappa, confusion_matrix),
)

# ---------------------------------------------------------------------------- #
#                              regression models                               #
# ---------------------------------------------------------------------------- #
model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav),
)

model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:randomforest),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav),
)

model, _, _ = symbolic_analysis(
    Xr, yr;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms, l1, l2, mae, mav),
)

# ---------------------------------------------------------------------------- #
#                    randomforest classification crash test                    #
# ---------------------------------------------------------------------------- #
@testset "randomforest data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:5:40
            for sampling_fraction in 0.7:0.1:0.9
                for n_trees in 10:10:100
                    model, mach, ds = symbolic_analysis(
                        Xc, yc;
                        model=(;type=:randomforest, params=(;n_trees, sampling_fraction)),
                        resample = (type=Holdout, params=(;shuffle=true)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(accuracy,),
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
                    mlj_acc = accuracy(yhat, ds.y[ds.tt[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                      randomforest regression crash test                      #
# ---------------------------------------------------------------------------- #
@testset "randomforest data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:5:40
            for sampling_fraction in 0.7:0.1:0.9
                for n_trees in 10:10:100
                    model, mach, ds = symbolic_analysis(
                        Xr, yr;
                        model=(;type=:randomforest, params=(;n_trees, sampling_fraction)),
                        resample = (type=Holdout, params=(;shuffle=true)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(rms,),
                    )
                    sx_rms = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
                    mlj_rms = rms(yhat, ds.y[ds.tt[1].test])

                    @test sx_rms == mlj_rms
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                              adaboost crash test                             #
# ---------------------------------------------------------------------------- #
@testset "data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:40
            for feature_importance in [:impurity, :split]
                for n_iter in 1:5:100
                    model, mach, ds = symbolic_analysis(
                        Xc, yc;
                        model=(type=:adaboost, params=(;n_iter, feature_importance)),
                        resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(seed))),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(accuracy, confusion_matrix),
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
                    mlj_acc = accuracy(yhat, ds.y[ds.tt[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                      xgboost classification crash test                       #
# ---------------------------------------------------------------------------- #
@testset "xgboost classification data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:5:40
            for num_round in 10:5:50
                for eta in 0.1:0.1:0.4
                    model, mach, ds = symbolic_analysis(
                        Xc, yc;
                        model=(;type=:xgboost, params=(;eta, num_round)),
                        resample = (type=Holdout, params=(;shuffle=true)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(accuracy,),
                    )
                    sx_acc = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
                    mlj_acc = accuracy(yhat, ds.y[ds.tt[1].test])

                    @test sx_acc == mlj_acc
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                        xgboost regression crash test                         #
# ---------------------------------------------------------------------------- #
@testset "xgboost data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:5:40
            for num_round in 10:5:50
                for eta in 0.1:0.1:0.4
                    model, mach, ds = symbolic_analysis(
                        Xr, yr;
                        model=(;type=:xgboost, params=(;eta, num_round)),
                        resample = (type=Holdout, params=(;shuffle=true)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(rms,),
                    )
                    sxhat = model.model[1].info.supporting_predictions
                    yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
                    sx_rms = model.measures.measures_values[1]
                    mlj_rms = rms(yhat, ds.y[ds.tt[1].test])

                    @test isapprox(sxhat, yhat; rtol=1e-6)
                    @test isapprox(sx_rms, mlj_rms; rtol=1e-6)
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                                  benchmarks                                  #
# ---------------------------------------------------------------------------- #
@btime begin
    model, _, _ = symbolic_analysis(
        Xc, yc;
        model=(;type=:xgboost),
        resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
        preprocess=(train_ratio=0.7, rng=Xoshiro(1)),
        measures=(accuracy,),
    )
end
# 7.188 ms (41827 allocations: 2.14 MiB)

@btime begin
    $(Xtrain, Xtest), $(ytrain, ytest) = partition((Xc, yc), 0.7, rng=Xoshiro(1), multi=true)
    Tree = @load XGBoostClassifier pkg=XGBoost
    tree = Tree()

    mach = machine(tree, Xc, yc) |> MLJ.fit!
    yhat = MLJ.predict_mode(mach, Xtest)
    meas = accuracy(yhat, ytest)
end
# 14.866 ms (14557 allocations: 935.15 KiB)

@btime begin
    model, _, _ = symbolic_analysis(
        Xr, yr;
        model=(;type=:xgboost),
        resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
        preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
        measures=(rms,),
    )
end
# 46.110 ms (553999 allocations: 20.85 MiB)

@btime begin
    model, ds = prepare_dataset(
        Xr, yr;
        model=(;type=:xgboost),
        resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
        preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
        measures=(rms,),
    )
end
# 30.197 μs (218 allocations: 73.41 KiB)

@btime begin
    model, mach, ds = train_test(
        Xr, yr;
        model=(;type=:xgboost),
        resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
        preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
        measures=(rms,),
    )
end
# 43.402 ms (553855 allocations: 20.83 MiB)

@btime begin
    $(Xtrain, Xtest), $(ytrain, ytest) = partition((Xr, yr), 0.7, rng=Xoshiro(1), multi=true)
    Tree = @load XGBoostRegressor pkg=XGBoost
    tree = Tree()

    mach = machine(tree, Xr, yr) |> MLJ.fit!
    yhat = MLJ.predict_mode(mach, Xtest)
    meas = MLJ.rms(yhat, ytest)
end
# 113.354 ms (14865 allocations: 1.15 MiB)

# ---------------------------------------------------------------------------- #
#                             remove logiset debug                             #
# ---------------------------------------------------------------------------- #
model, ds = prepare_dataset(
    Xr, yr;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms,)
)

const SD = SoleData
X = ds.X
@btime dX = DataFrame(ds.X, ds.info.vnames)
# 10.110 μs (61 allocations: 51.47 KiB)
@btime mX = MLJ.table(ds.X; names=ds.info.vnames)
# 1.249 μs (14 allocations: 1.30 KiB)

dX = DataFrame(ds.X, ds.info.vnames)
mX = MLJ.table(ds.X; names=ds.info.vnames)

dS = SD.scalarlogiset(dX; silent=true, allow_propositional = true);
mS = SD.scalarlogiset(mX; silent=true, allow_propositional = true);

@btime SD.scalarlogiset(dX; silent=true, allow_propositional = true);
# 5.397 ms (24826 allocations: 26.35 MiB)
@btime SD.scalarlogiset(mX; silent=true, allow_propositional = true);
# 4.079 ms (20282 allocations: 24.25 MiB)

model, ds = prepare_dataset(
    Xts, yts;
    model=(;type=:modaldecisiontree),
    resample = (type=Holdout, params=(shuffle=true, rng=Xoshiro(1))),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(rms,)
)

a = false
@btime a === false ? 1 : 0
# 2.815 ns (0 allocations: 0 bytes)
@btime a ? 1 : 0
# 3.265 ns (0 allocations: 0 bytes)
@btime !a ? 1 : 0
# 14.892 ns (0 allocations: 0 bytes)

dX = DataFrame(ds.X, ds.info.vnames)
mX = MLJ.table(ds.X; names=ds.info.vnames)

dS = SD.scalarlogiset(dX)
mS = SD.scalarlogiset(mX)

SD.matrix2dimensional(mX)