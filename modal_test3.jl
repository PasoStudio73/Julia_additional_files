using Test
using MLJ
using SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

a = prepare_dataset(
        Xts, yts;
        model=DecisionTreeClassifier(),
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
)

@btime begin
    a = prepare_dataset(
        Xts, yts;
        model=DecisionTreeClassifier(),
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
    )
end

#######################################################################
X = Xts
win=AdaptiveWindow(nwindows=3, relative_overlap=0.1)
# win = WholeWindow()
features=[maximum, minimum]
treat=:reducesize
modalreduce=mean

######################################################################
vnames = propertynames(X)

# run the windowing algo and set windows indexes
intervals = win(length(X[1,1]))
n_intervals = length(intervals)

######################################################################
win = WholeWindow()
intervals = win(length(X[1,1]))
n_intervals = length(intervals)

@btime begin
    _X = DataFrame(
        [Symbol(f, "(", v, ")") => [f(ts) for ts in X[!, v]]
            for f in features
            for v in vnames]...)
end
# 4.490 ms (1005 allocations: 315.88 KiB)

# Ultimate optimization: Custom broadcast function
function apply_features_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    feature_func::Function,
    col_name::Symbol
)
    X[!, col_name] = collect(feature_func(X_col[i]) for i in 1:length(X_col))
end

@btime begin
    _X = DataFrame()
    for f in features
        @simd for v in vnames
            col_name = Symbol(f, "(", v, ")")
            apply_features_vectorized!(_X, X[!, v], f, col_name)
        end
    end
end
# 4.426 ms (509 allocations: 166.80 KiB)

######################################################################
win=AdaptiveWindow(nwindows=3, relative_overlap=0.1)
intervals = win(length(X[1,1]))
n_intervals = length(intervals)

@btime begin
    _X = DataFrame((
        Symbol(f, "(", v, ")w", i) => f.(map(v -> @view(v[interval]), X[!, v]))
        for f in features
        for v in vnames
        for (i, interval) in enumerate(intervals)
    )...)
end
# 7.074 ms (4317 allocations: 2.93 MiB)

@btime begin
    _X = DataFrame()
    for f in features
        @simd for v in vnames
            for (i, interval) in enumerate(intervals)
                col_name = Symbol(f, "(", v, ")w", i)
                _X[!, col_name] = collect(f(@views ts[interval]) for ts in X[!, v])
            end
        end
    end
end
# 6.134 ms (2813 allocations: 524.73 KiB)

function apply_features_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    feature_func::Function,
    col_name::Symbol,
    interval::UnitRange{Int64}
)
    X[!, col_name] = collect(feature_func(@views X_col[i][interval]) for i in 1:length(X_col))
end

@btime begin
    _X = DataFrame()
    for f in features
        @simd for v in vnames
            for (i, interval) in enumerate(intervals)
                col_name = Symbol(f, "(", v, ")w", i)
                apply_features_vectorized!(_X, X[!, v], f, col_name, interval)
            end
        end
    end
end
# 6.322 ms (2525 allocations: 515.73 KiB)