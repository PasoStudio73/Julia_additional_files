using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = load_arff_dataset("NATOPS")

# I'm easy like sunday morning
modelc = symbolic_analysis(Xc, yc)
@test modelc isa SX.ModelSet

# ---------------------------------------------------------------------------- #
#                               usage example #1                               #
# ---------------------------------------------------------------------------- #
range = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)
dsc = setup_dataset(
    Xc, yc;
    model=XGBoostClassifier(),
    resample=CV(nfolds=5, shuffle=true),
    rng=Xoshiro(1),
    tuning=(tuning=Grid(resolution=10), resampling=CV(nfolds=3), range, measure=accuracy, repeats=2)    
)
solemc = train_test(dsc)
modelc = symbolic_analysis(
    dsc, solemc;
    # extractor=InTreesRuleExtractor(),
    measures=(accuracy, log_loss, confusion_matrix, kappa)
)
@test modelc isa SX.ModelSet

function _symbolic_analysis(
    ds::EitherDataSet,
    solem::SModel;
    extractor::Union{Nothing,RuleExtractor}=nothing,
    measures::Tuple{Vararg{FussyMeasure}}=(),
)::ModelSet
    rules = isnothing(extractor)  ? nothing : begin
        # TODO propaga rng, dovrai fare intrees mutable struct
        extractrules(extractor, ds, solem)
    end

    measures = isempty(measures) ? nothing : begin
        y_test = get_y_test(ds)
        # all_classes = unique(Iterators.flatten(y_test))
        eval_measures(ds, solem, measures, y_test)
    end

    return ModelSet(ds, solem; rules, measures)
end

dsc = symbolic_analysis(
    Xc, yc,
    model=SX.XGBoostClassifier(),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy,)
)

dsr = symbolic_analysis(
    Xr, yr,
    model=SX.XGBoostRegressor(),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(rms,)
)

@btime begin
    dsc = symbolic_analysis(
    Xc, yc,
    model=SX.XGBoostClassifier(),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy,)
)
end
# 383.803 ms (2013184 allocations: 107.30 MiB)
# 387.927 ms (2187854 allocations: 107.35 MiB)

@btime begin
dsr = symbolic_analysis(
    Xr, yr,
    model=SX.XGBoostRegressor(),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(rms,)
)
end
# 2.888 s (22959657 allocations: 903.24 MiB)

@btime begin
    Tree = @load XGBoostClassifier pkg=XGBoost verbosity=0
    tree = Tree()
    evaluate(
        tree, Xc, yc;
        resampling=CV(nfolds=10, shuffle=true),
        measures=[accuracy,],
        per_observation=true,
        verbosity=0
    )
end
# 82.527 ms (16956 allocations: 1.08 MiB)

@btime begin
    Tree = @load XGBoostRegressor pkg=XGBoost verbosity=0
    tree = Tree()
    evaluate(
        tree, Xr, yr;
        resampling=CV(nfolds=10, shuffle=true),
        measures=[rms,],
        per_observation=true,
        verbosity=0
    )
end
# 986.137 ms (14683 allocations: 2.56 MiB)

# ---------------------------------------------------------------------------- #
#                       Sole vs MLJ machine & fit setup                        #
# ---------------------------------------------------------------------------- #
dsc = symbolic_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

@btime begin
    symbolic_analysis(
        Xc, yc,
        model=DecisionTreeClassifier(),
        resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
        measures=(accuracy, kappa)
    )
end
# 468.033 μs (3446 allocations: 250.73 KiB)

@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    tree = Tree()
    evaluate(
        tree, Xc, yc;
        resampling=Holdout(shuffle=true),
        measures=[accuracy, kappa],
        per_observation=true,
        verbosity=0
    )
end
# 394.518 μs (1900 allocations: 118.90 KiB)

@btime begin
    symbolic_analysis(
        Xr, yr,
        model=SX.RandomForestRegressor(n_trees=100),
        resample=CV(nfolds=10, shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
        measures=(rms,)
    )
end
# 4.882 s (30678086 allocations: 1.27 GiB)
# 1.394 s (19436815 allocations: 653.31 MiB)
# senza apply
# 1.056 s (10897211 allocations: 398.17 MiB)

# rifaccio DecisionTreeExt
# 1.553 s (19437247 allocations: 653.34 MiB)
# tipizzazione
# 1.587 s (19436765 allocations: 653.31 MiB)
# alleggerito il primo apply
# 1.477 s (19380393 allocations: 645.15 MiB)
# 936.445 ms (11692285 allocations: 461.07 MiB)
# 892.104 ms (10975930 allocations: 437.26 MiB)

@btime begin
    Tree = @load RandomForestRegressor pkg=DecisionTree verbosity=0
    tree = Tree(n_trees=100)
    evaluate(
        tree, Xr, yr;
        resampling=CV(nfolds=10,shuffle=true),
        measures=rms,
        per_observation=true,
        verbosity=0
    )
end
# 484.860 ms (1823896 allocations: 161.90 MiB)

@btime begin
    ds = setup_dataset(
        Xr, yr,
        model=RandomForestRegressor(n_trees=100),
        resample=CV(nfolds=10, shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
    )
    model = train_test(ds)
end
# 4.949 s (30675782 allocations: 1.27 GiB)

# ---------------------------------------------------------------------------- #
#                              SModel bench                                 #
# ---------------------------------------------------------------------------- #
ds = setup_dataset(
    Xr, yr,
    model=RandomForestRegressor(n_trees=100),
    resample=CV(nfolds=10, shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
)

get_X(ds::SX.AbstractDataSet)::DataFrame = ds.mach.args[1].data
get_y(ds::SX.AbstractDataSet)    = @views ds.mach.args[2].data

# function _train_test(ds::EitherDataSet)
n_folds   = length(ds.pidxs)
solemodel = Vector{SX.AbstractModel}(undef, n_folds)

@inbounds @views for i in 1:n_folds
    train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
    X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

    SX.has_xgboost_model(ds) && SX.set_watchlist!(ds, i)

    MLJ.fit!(ds.mach, rows=train, verbosity=0)
    solemodel[i] = SX.apply(ds, X_test, y_test)
end

@btime begin
    @inbounds @views for i in 1:n_folds
        train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

        SX.has_xgboost_model(ds) && SX.set_watchlist!(ds, i)

        MLJ.fit!(ds.mach, rows=train, verbosity=0)
        solemodel[i] = SX.apply(ds, X_test, y_test)
    end
end
# 4.083 s (30724331 allocations: 1.27 GiB)
# senza apply 456.737 ms (1821494 allocations: 161.18 MiB)

ds = setup_dataset(
    Xc, yc,
    model=XGBoostClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
)
i = 1
train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
X_test, y_test = SX.get_X(ds)[test, :], SX.get_y(ds)[test]
MLJ.fit!(ds.mach, rows=train, verbosity=0)
trees        = SX.XGBoost.trees(ds.mach.fitresult[1])
encoding     = SX.get_encoding(ds.mach.fitresult[2])
classlabels  = string.(SX.get_classlabels(encoding))
featurenames = ds.mach.report.vals[1].features
solem        = SX.solemodel(trees, X_test, y_test; classlabels, featurenames)
propositional_apply!(solem, mapcols(col -> Float32.(col), X), y)
