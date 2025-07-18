using Test
using SoleXplorer
using MLJ, DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = load_arff_dataset("NATOPS")

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

solemc = train_test(dsc)

@btime train_test($dsc)
# 263.497 μs (2791 allocations: 206.25 KiB)

fit!(mach, rows=train)
yhat = predict(mach, Xc[test,:])

@btime begin
    fit!($mach, rows=$train, force=true)
    yhat = predict($mach, $Xc[test,:])
end
# 121.420 μs (1130 allocations: 73.84 KiB)

modelc = symbolic_analysis(dsc, solemc, measures=(accuracy,))

@btime symbolic_analysis($dsc, $solemc, measures=(accuracy,))
# 9.234 μs (120 allocations: 5.70 KiB)

accuracy(yhat, yc[test])

@btime accuracy($yhat, $yc[test])
# 8.399 μs (433 allocations: 21.23 KiB)

# ---------------------------------------------------------------------------- #
#                             Sole vs MLJ evaluate                             #
# ---------------------------------------------------------------------------- #
@btime begin
    dsc = setup_dataset(
        $Xc, $yc,
        model=$DecisionTreeClassifier(),
        resample=(type=$Holdout(shuffle=true), train_ratio=0.7, rng=$Xoshiro(1))
    )
    solemc = train_test(dsc)
    symbolic_analysis(dsc, solemc, measures=(accuracy,))
end
# 414.535 μs (3324 allocations: 246.18 KiB)

parziali_t = 18.171 + 263.497 + 9.234
parziali_a = 145 + 2791 + 120
parziali_m = 8.97 + 206.25 + 5.70

Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
tree = Tree()
mljc = evaluate(
    tree, Xc, yc,
    resampling=Holdout(shuffle=true),
    measures=[accuracy],
    verbosity=0
)

@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    tree = Tree()
    mljc = evaluate(
        tree, $Xc, $yc,
        resampling=$Holdout(shuffle=true),
        measures=[accuracy],
        verbosity=0
    )
end
# 308.983 μs (1736 allocations: 110.99 KiB)
