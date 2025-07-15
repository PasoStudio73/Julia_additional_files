using Test
using MLJ, SoleXplorer
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

@btime begin
    model = train_test(
        Xc, yc;
        model=(;type=:decisiontree)
    )
end
# main   - 329.832 μs (1964 allocations: 182.55 KiB)

@btime begin
    model, mach, ds = train_test(
        Xc, yc;
        model=(;type=:decisiontree)
    )
end
# 67-dev - 307.978 μs (1960 allocations: 182.33 KiB)

@btime begin
    m = symbolic_analysis(
        Xc, yc;
        model=(;type=:decisiontree)
    )
end
# main   - 478.555 μs (2867 allocations: 235.17 KiB)
# 67-dev - 468.925 μs (2851 allocations: 234.77 KiB)

# Xtrain = @views ds.X[ds.tt[1].train, :]
# ytrain = @views ds.y[ds.tt[1].train]
# Xtest  = @views ds.X[ds.tt[1].test,  :]
# ytest  = @views ds.y[ds.tt[1].test ]

# ---------------------------------------------------------------------------- #
#                         check train test partitions                          #
# ---------------------------------------------------------------------------- #
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

@test_nowarn begin
    for seed in 1:25
        for fraction_train in 0.5:0.1:0.9
            Random.seed!(1234)
            _, ds = setup_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                # don't pass fraction_ratio to resample, it goes to preprocess
                resample = (type=Holdout, params=(;shuffle=true)),
                preprocess=(train_ratio=fraction_train, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=Holdout(;fraction_train, shuffle=true, rng=Xoshiro(seed)),
                per_observation=false,
                verbosity=0,
            )
            @assert ds.tt[1].test == mljm.train_test_rows[1][2]
        end
    end
end

@test_nowarn begin
    for seed in 1:25
        for nfolds in 2:25
            Random.seed!(1234)
            _, ds = setup_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                resample = (type=CV, params=(;nfolds, shuffle=true)),
                preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=CV(;nfolds, shuffle=true, rng=Xoshiro(seed)),
                per_observation=false,
                verbosity=0,
            )

            @assert all(ds.tt[i].test == mljm.train_test_rows[i][2] for i in 1:nfolds)
        end
    end
end

@test_nowarn begin
    for seed in 1:25
        for nfolds in 2:25
            Random.seed!(1234)
            _, ds = setup_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                resample = (type=StratifiedCV, params=(;nfolds, shuffle=true)),
                preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=StratifiedCV(;nfolds, shuffle=true, rng=Xoshiro(seed)),
                per_observation=false,
                verbosity=0,
            )

            @assert all(ds.tt[i].test == mljm.train_test_rows[i][2] for i in 1:nfolds)
        end
    end
end

@test_nowarn begin
    for seed in 1:25
        for nfolds in 2:25
            Random.seed!(1234)
            _, ds = setup_dataset(
                Xc, yc;
                model=(;type=:decisiontree),
                resample = (type=TimeSeriesCV, params=(;nfolds)),
                preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
            )

            Random.seed!(1234)
            mljm = evaluate(
                tree, Xc, yc;
                resampling=TimeSeriesCV(;nfolds),
                per_observation=false,
                verbosity=0,
            )

            @assert all(ds.tt[i].test == mljm.train_test_rows[i][2] for i in 1:nfolds)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                             check decision trees                             #
# ---------------------------------------------------------------------------- #
seed = 1
nfolds = 6

Random.seed!(1234)
solem, _, _ = train_test(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=CV, params=(;nfolds, shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
)

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
Random.seed!(1234)
mljm = evaluate(
    tree, Xc, yc;
    resampling=CV(;nfolds, shuffle=true, rng=Xoshiro(seed)),
    per_observation=false,
    verbosity=0,
)

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
Random.seed!(1234)
_, ds = setup_dataset(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=CV, params=(;nfolds, shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(seed)),
)
mach = machine(tree, Xc, yc)
fit!(mach, rows=ds.tt[3].train)
