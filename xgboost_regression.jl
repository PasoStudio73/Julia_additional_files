using Test
using MLJ, SoleXplorer
using DataFrames, Random
using MLJXGBoostInterface
using SoleModels
using XGBoost
const SX = SoleXplorer
const XGB = XGBoost

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#                          classification crash test                           #
# ---------------------------------------------------------------------------- #
@testset "data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:40
            _, ds = prepare_dataset(Xc, yc; preprocess=(;train_ratio, rng=Xoshiro(seed)))
            X_train, y_train = ds.X[ds.tt[1].train, :], ds.y[ds.tt[1].train]
            X_test, y_test = ds.X[ds.tt[1].test, :], ds.y[ds.tt[1].test]

            for num_round in 10:10:50
                for eta in 0.1:0.1:0.6
                    # XGBoost model
                    yl_train = MLJ.levelcode.(MLJ.categorical(y_train)) .- 1
                    bst = XGB.xgboost((X_train, yl_train); num_round, eta, num_class=3, objective="multi:softmax", verbosity=0)
                    xg_preds = XGB.predict(bst, X_test)
                    yl_test = MLJ.levelcode.(MLJ.categorical(y_test)) .- 1
                    xg_accuracy = sum(xg_preds .== yl_test) / length(yl_test)

                    # SoleXplorer model
                    model = symbolic_analysis(
                        Xc, yc;
                        model=(type=:xgboost, params=(;num_round, eta)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed)),
                        measures=(accuracy,)
                    )

                    @test model.measures.measures_values[1] == xg_accuracy
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                          classification crash test                           #
# ---------------------------------------------------------------------------- #
@testset "data validation" begin
    for train_ratio in 0.5:0.1:0.9
        for seed in 1:40
            _, ds = prepare_dataset(Xr, yr; preprocess=(;train_ratio, rng=Xoshiro(seed)))
            X_train, y_train = ds.X[ds.tt[1].train, :], ds.y[ds.tt[1].train]
            X_test, y_test = ds.X[ds.tt[1].test, :], ds.y[ds.tt[1].test]

            for num_round in 10:10:50
                for eta in 0.1:0.1:0.6
                    # XGBoost model
                    bst = XGB.xgboost((X_train, yl_train); num_round, eta, num_class=3, objective="multi:softmax", verbosity=0)
                    xg_preds = XGB.predict(bst, X_test)

                    # SoleXplorer model
                    model = train_test(
                        Xr, yr;
                        model=(type=:xgboost, params=(;num_round, eta)),
                        preprocess=(;train_ratio, rng=Xoshiro(seed))
                    )

                    @test model.measures.measures_values[1] == xg_accuracy
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                     build xgboost regressor from scratch                     #
# ---------------------------------------------------------------------------- #
model, ds = prepare_dataset(Xr, yr; model=(;type=:xgboost), preprocess=(;train_ratio=0.7, rng=Xoshiro(1)))
X_train, y_train = ds.X[ds.tt[1].train, :], ds.y[ds.tt[1].train]
X_test, y_test = ds.X[ds.tt[1].test, :], ds.y[ds.tt[1].test]

params = (;num_round=2, max_depth=3, objective="reg:squarederror", base_score=mean(y_train))
params = (;num_round=2, max_depth=3, objective="reg:squarederror", base_score=-Inf)
# julia XGBoost package
bst  = XGB.xgboost((X_train, y_train); params...)
tree = XGB.trees(bst)
xg_yhat = XGB.predict(bst, X_test)

# SoleXplorer train
mach = MLJ.machine(
    MLJXGBoostInterface.XGBoostRegressor(;params...),
    MLJ.table(ds.X; names=ds.info.vnames),
    ds.y
)

# SoleXplorer test
MLJ.fit!(mach, rows=ds.tt[1].train, verbosity=0)

# SoleXplorer apply
X_test  = DataFrame((@views ds.X[ds.tt[1].test, :]), ds.info.vnames)
sxtrees      = XGB.trees(mach.fitresult[1])
featurenames = mach.report.vals[1].features
solem        = solemodel(sxtrees, Matrix(X_test), y_test; featurenames)

# SoleModels variablenames
d = mapcols(col -> Float32.(col), X_test)
silent = false
d = SoleData.scalarlogiset(d; silent, allow_propositional = true)
m = solem
y = Float32.(y_test)
suppress_parity_warning = false
mode = :replace
leavesonly = false

# SoleModels apply
y = SoleModels.__apply_pre(m, d, y)
preds_step1 = hcat([SoleModels.apply!(subm, d, y; mode, leavesonly) for subm in SoleModels.models(m)]...)
preds_step2 = SoleModels.__apply_post(m, preds_step1)
preds_step3 = [SoleModels.aggregation(m)(p) for p in eachrow(preds_step2)]
preds_step4 = SoleModels.__apply_pre(m, d, preds_step3)
preds_step5 = SoleModels.__apply!(m, mode, preds_step4, y, leavesonly)
preds_step6 = (preds_step5*2) .+ params.base_score

### TO FIX
# function test_bestguess(
#     labels::AbstractVector{<:SoleModels.RLabel},
#     weights::Union{Nothing, AbstractVector} = nothing;
#     suppress_parity_warning = false,
# )
#     if length(labels) == 0
#         return nothing
#     end

#     (isnothing(weights) ? StatsBase.mean(labels) : sum(labels .* weights) / sum(weights))
# end


# test_aggregation = function(args...) test_bestguess(args...) end

testpreds = [SoleModels.aggregation(m)(p) for p in eachrow(preds)]

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
"""
Calculate mean squared error
"""
function mse(y_true::Vector{<:AbstractFloat}, y_pred::Vector{<:AbstractFloat})
    return mean((Float32.(y_true) .- Float32.(y_pred)).^2)
end

"""
Calculate R²
"""
function r2_score(y_true::Vector{<:AbstractFloat}, y_pred::Vector{<:AbstractFloat})
    ss_res = sum((Float32.(y_true) .- Float32.(y_pred)).^2)
    ss_tot = sum((Float32.(y_true) .- mean(Float32.(y_true))).^2)
    return 1 - (ss_res / ss_tot)
end
# ---------------------------------------------------------------------------- #
#                             various benchmarks                               #
# ---------------------------------------------------------------------------- #
@btime MLJ.machine(
    MLJXGBoostInterface.XGBoostRegressor(;num_round=2, max_depth=3, objective="reg:squarederror"),
    MLJ.table(@views ds.X; names=ds.info.vnames),
    @views ds.y
)
# 11.762 μs (65 allocations: 3.91 KiB)

@btime MLJ.machine(
    MLJXGBoostInterface.XGBoostRegressor(;num_round=2, max_depth=3, objective="reg:squarederror"),
    MLJ.table(@views ds.X; names=ds.info.vnames),
    ds.y
)
# 11.761 μs (65 allocations: 3.91 KiB)

@btime MLJ.machine(
    MLJXGBoostInterface.XGBoostRegressor(;num_round=2, max_depth=3, objective="reg:squarederror"),
    MLJ.table(ds.X; names=ds.info.vnames),
    ds.y
)
# 11.764 μs (65 allocations: 3.91 KiB)

@btime MLJ.table(ds.X; names=ds.info.vnames)
# 1.247 μs (14 allocations: 1.30 KiB)

@btime @views MLJ.table(ds.X; names=ds.info.vnames)
# 1.246 μs (14 allocations: 1.30 KiB)

@btime a = (@views(Matrix(ds.X)), @views(ds.y))
# 5.007 μs (6 allocations: 47.71 KiB)

@btime a = (Matrix(ds.X), ds.y)
# 4.910 μs (6 allocations: 47.71 KiB)

@btime hcat([SoleModels.apply!(subm, d, y; mode, leavesonly) for subm in SoleModels.models(m)]...)
# 236.506 μs (2703 allocations: 155.06 KiB)

@btime first.(hcat([SoleModels.apply_leaf_scores(subm, d; suppress_parity_warning) for subm in SoleModels.models(m)]...))
# 303.998 μs (3378 allocations: 167.49 KiB)

@btime begin
    y = SoleModels.__apply_pre($m, $d, $y)
    preds = hcat([SoleModels.apply!(subm, d, y; mode, leavesonly) for subm in SoleModels.models(m)]...)
    preds = SoleModels.__apply_post(m, preds)
    preds = [SoleModels.aggregation(m)(p) for p in eachrow(preds)]
    preds = SoleModels.__apply_pre(m, d, preds)
    preds = SoleModels.__apply!(m, mode, preds, y, leavesonly)
    preds = (preds*2) .+ params.base_score
end
# 245.770 μs (3026 allocations: 163.92 KiB)

@btime final_preds = let
    models = SoleModels.models(m)
    aggregation_fn = SoleModels.aggregation(m)
    
    # Define pipeline functions
    preprocess = y_data -> SoleModels.__apply_pre(m, d, y_data)
    apply_submodels = y_prep -> reduce(hcat, (SoleModels.apply!(subm, d, y_prep; mode, leavesonly) for subm in models))
    post_process = preds -> SoleModels.__apply_post(m, preds)
    aggregate = processed -> map(aggregation_fn, eachrow(processed))
    final_transform = (aggregated, y_prep) -> begin
        SoleModels.__apply_pre(m, d, aggregated) |>
        (preprocessed -> SoleModels.__apply!(m, mode, preprocessed, y_prep, leavesonly)) |>
        (applied -> @. 2 * applied + params.base_score)
    end
    
    # Execute pipeline
    y_preprocessed = y |> preprocess
    y_preprocessed |> apply_submodels |> post_process |> aggregate |> 
    (aggregated -> final_transform(aggregated, y_preprocessed))
end

model, ds = train_test(
    Xc, yc;
    model=(;type=:xgboost),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1))
)

model, ds = train_test(
    Xr, yr;
    model=(type=:xgboost, params=(;num_round=2, max_depth=3, objective="reg:squarederror")),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1))
)

model = symbolic_analysis(
    Xc, yc;
    model=(;type=:xgboost),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(specificity,)
)

model = symbolic_analysis(
    Xr, yr;
    model=(type=:xgboost, params=(;num_round=2, max_depth=3, objective="reg:squarederror")),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    measures=(mae,)
)