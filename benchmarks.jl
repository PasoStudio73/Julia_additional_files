using Test
using SoleXplorer
using MLJ, DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

# Xts, yts = load_arff_dataset("NATOPS")

@btime begin
    evaluate(
        tree, Xr, yr;
        resampling=CV(nfolds=10,shuffle=true),
        measures=rms,
        per_observation=true,
        verbosity=0,
        cache=true
    )
end
# 523.425 ms (1825099 allocations: 161.95 MiB)
@btime begin
    evaluate(
        tree, Xr, yr;
        resampling=CV(nfolds=10,shuffle=true),
        measures=rms,
        per_observation=true,
        verbosity=0,
        cache=false
    )
end
# 544.962 ms (1823246 allocations: 162.35 MiB)

y=SX.get_y(ds)
@btime nrows = MLJ.MLJBase.nrows(y)
# 14.929 ns (0 allocations: 0 bytes)
@btime nrows = length(y)
# 17.043 ns (0 allocations: 0 bytes)

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
        model=RandomForestRegressor(n_trees=100),
        resample=CV(nfolds=10, shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),
        measures=(rms,)
    )
end
# 4.882 s (30678086 allocations: 1.27 GiB)

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
#                              SoleModel bench                                 #
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

# function apply
i = 1
train, test = SX.get_train(ds.pidxs[i]), SX.get_test(ds.pidxs[i])
X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

featurenames = MLJ.report(ds.mach).features
solem        = SX.solemodel(MLJ.fitted_params(ds.mach).forest; featurenames)

dsr = symbolic_analysis(
    Xr, yr,
    model=RandomForestRegressor(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(rms,)
)