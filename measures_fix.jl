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
                    sx_rms = model.measures.measures_values[1]
                    yhat = MLJ.predict_mode(mach, MLJ.table(ds.X[ds.tt[1].test, :]))
                    mlj_rms = rms(yhat, ds.y[ds.tt[1].test])

                    @show train_ratio, seed, num_round, eta
                    @test sx_rms == mlj_rms
                end
            end
        end
    end
end
