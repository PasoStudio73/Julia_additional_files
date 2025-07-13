using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")


# ---------------------------------------------------------------------------- #
Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0;
tree = Tree()

model = DecisionTreeRegressor()

r1 = SX.range(:min_purity_increase; lower=0.001, upper=1.0, scale=:log)

r2 = (SX.range(:min_purity_increase, lower=0.001, upper=1.0, scale=:log),
     SX.range(:max_depth, lower=1, upper=10))
     
r1m = MLJ.range(model, r1[1]; r1[2:end]...)

r2m = collect(MLJ.range(model, r[1]; r[2:end]...) for r in r2)

self_tuning_tree = TunedModel(
    model=DecisionTreeRegressor(),
    resampling=CV(nfolds=3),
    tuning=Grid(resolution=10),
    range=r1,
    measure=(l1, rms)
)

mach = machine(self_tuning_tree, Xr, yr)
fit!(mach, verbosity=0)

modelr = prepare_dataset(
    Xr, yr;
    model,
    resample=(;rng=Xoshiro(1234)),
    tuning=(tuning=Grid(resolution=10),resampling=CV(nfolds=3),range=r1,measure=rms)
)

modelc = prepare_dataset(
    Xc, yc;
    model,
    resample=(;rng=Xoshiro(1234)),
    tuning=(tuning=Grid(resolution=10),resampling=CV(nfolds=3),range=r2,measure=rms)
)

tuning=(tuning=Grid(resolution=10),resampling=CV(nfolds=3),range=r1,measure=rms)
tuning=(tuning=Grid(resolution=10),resampling=CV(nfolds=3),range=r2,measure=rms)

Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0;
tree = Tree()

model = DecisionTreeRegressor()

selector = FeatureSelector();
r2 = MLJ.range(selector, :features, values = [[:sepal_width,], [:sepal_length, :sepal_width]])

self_tuning_tree = TunedModel(
    model=DecisionTreeRegressor(),
    resampling=CV(nfolds=3),
    tuning=Grid(resolution=10),
    range=r2,
    measure=(l1, rms)
)

modelc = prepare_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=(;rng=Xoshiro(1234)),
    tuning=(tuning=Grid(resolution=10),resampling=CV(nfolds=3),range=r2,measure=rms)
)

