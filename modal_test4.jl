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
        win=WholeWindow(),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
    )
end
# 4.422 ms (568 allocations: 176.79 KiB)

@btime begin
    a = prepare_dataset(
        Xts, yts;
        model=DecisionTreeClassifier(),
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
    )
end
# 6.372 ms (2271 allocations: 510.84 KiB)

@btime begin
    a = prepare_dataset(
        Xts, yts;
        model=ModalDecisionTree(),
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
    )
end
# 710.409 μs (17591 allocations: 764.54 KiB)

#######################################################################
X = Xts
features=[maximum, minimum]
treat=:reducesize
modalreduce=mean
vnames = propertynames(X)


######################################################################
win = WholeWindow()
intervals = win(length(X[1,1]))
n_intervals = length(intervals)

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

######################################################################
@btime begin
    _X = DataFrame([
        col => map(v -> modalreduce.(@views v[i] for i in intervals), X[!, col])
        for col in vnames
    ])
end
# 3.547 ms (60871 allocations: 2.78 MiB)

@btime begin
    _X = DataFrame()
    
    @simd for v in vnames
        for interval in intervals
            _X[!, v] = map(x -> modalreduce.(@views x[i] for i in intervals), X[!, col])
        end
    end
end

####################################################################################

function apply_features_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    modalreduce_func::Function,
    col_name::Symbol,
    intervals::Vector{UnitRange{Int64}},
    n_rows::Int,
    n_intervals::Int
)::Nothing
    result_column = Vector{Vector{Float64}}(undef, n_rows)
    row_result = Vector{Float64}(undef, n_intervals)
    
    @inbounds @fastmath for row_idx in 1:n_rows
        ts = X_col[row_idx]
        
        for (i, interval) in enumerate(intervals)
            row_result[i] = modalreduce_func(@view(ts[interval]))
        end
        result_column[row_idx] = copy(row_result)
    end

    X[!, col_name] = result_column
    
    return nothing
end

n_rows = nrow(X)

@btime begin
    _X = DataFrame()
    
    for v in vnames
        apply_features_vectorized!(_X, X[!, v], modalreduce, v, intervals, n_rows, n_intervals)
    end
end
# 561.605 μs (17394 allocations: 747.72 KiB)
