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

resample=(type=Holdout(shuffle=true), train_ratio=0.7, valid_ratio=0.1, rng=Xoshiro(1))
a=SX.partition(yc; resample...)

#######################################################################
X = Xts
# win=AdaptiveWindow(nwindows=3, relative_overlap=0.1)
win = WholeWindow()
features=[maximum, minimum]
treat=:reducesize
modalreduce=mean

######################################################################
vnames = propertynames(X)

# run the windowing algo and set windows indexes
intervals = win(length(first(X)))
n_intervals = length(intervals)


# define column names and prepare data structure based on treatment type
if treat == :aggregate        # propositional
    if n_intervals == 1
        # Apply feature to whole time series
        _X = DataFrame(
            [Symbol(f, "(", v, ")") => [f(ts) for ts in X[!, v]]
                for f in features
                for v in vnames]...)
        # _X = DataFrame(pairs...)  # Add the splat operator!
    else
        # apply feature to specific intervals
        _X = DataFrame(
            Symbol(f, "(", v, ")w", i) => f.(getindex.(X[!, v], Ref(interval)))
            for f in features
            for v in vnames
            for (i, interval) in enumerate(intervals)
        )
    end

elseif treat == :reducesize   # modal
    _X = DataFrame(
        [v => [
            [modalreduce(ts[interval]) for interval in intervals]
            for ts in X[!, v]
        ]
        for v in vnames]...
    )

else
    error("Unknown treatment type: $treat")
end

########################################################################

    result_df = DataFrame([col_name => Vector{Float64}(undef, nrows(X)) for col_name in col_names])
    for (row_idx, row) in enumerate(eachrow(X))
    
            # calculate feature values for this row
            feature_values = vcat([
            vcat([f(col[r]) for r in intervals]) for col in row, f in features
            ]...)
            result_df[row_idx, :] = feature_values
    end

    cX = DataFrame([col_name => Vector{Float64}(undef, nrows(X)) for col_name in col_names])
    col_idx = 1
    for f in features, v in vnames, interval in intervals
        cX[!, col_idx] = [f(X[row_idx, v][interval]) for row_idx in 1:nrows(X)]
        col_idx += 1
    end

    result_df == cX

    # build the exact (f, v, interval) ordering you used for col_names
    triples = collect(product(intervals, vnames, features))
    # build a Vector{Pair{Symbol,Vector{Float64}}}, one Pair per output‐column
    pairs = [col_names[i] => 
            map(x -> f(x[interval]), X[!, v])
            for (i, (interval, v, f)) in enumerate(triples)]
    _X = DataFrame(pairs...)

    result_df == _X



# return result_matrix, col_names

@btime propertynames(X)
# 44.853 ns (2 allocations: 256 bytes)
@btime names(X)
# 440.763 ns (26 allocations: 1.00 KiB)

using IterTools  # Optional, for product

function winfeats2dataframe(X, features, vnames, intervals)
    # triples = collect(product(features, vnames, intervals))
    # 2) Build a Vector{Pair{Symbol,Vector{Float64}}}, one Pair per output‐column
    pairs = [col_names[i] => 
            map(x -> f(x[interval]), X[!, v])
            for (i, (f, v, interval)) in enumerate(product(features, vnames, intervals))]

    # 3) Splat them into DataFrame so each Pair becomes its own column:
    DataFrame(pairs...)
end

@btime _X = winfeats2dataframe(X, features, vnames, intervals)
# 1) build the same ordered list of (f, v, interval) as your col_names

##############################################################################
@btime begin
    # build the exact (f, v, interval) ordering you used for col_names
    triples = collect(product(intervals, vnames, features))
    # build a Vector{Pair{Symbol,Vector{Float64}}}, one Pair per output‐column
    pairs = [col_names[i] => 
            map(x -> f(x[interval]), X[!, v])
            for (i, (interval, v, f)) in enumerate(triples)]
    _X = DataFrame(pairs...)
end
# 5.003 ms (105743 allocations: 7.46 MiB)



# Method 3: Ultra-fast matrix-based approach


# Method 4: Using broadcast for maximum speed
@btime begin
    _X = DataFrame(
        Symbol(f, "(", v, ")w", i) => f.(getindex.(X[!, v], Ref(interval)))
        for f in features
        for v in vnames
        for (i, interval) in enumerate(intervals)
    )
end

#############################################################

    # elseif treat == :reducesize   # modal
# col_names = vnames

n_rows = size(X, 1)
n_cols = length(col_names)
result_matrix = Matrix{T}(undef, n_rows, n_cols)

# modalreduce === nothing && (modalreduce = mean)

for (row_idx, row) in enumerate(eachrow(X))
    # row_intervals = winparams.type(maximum(length.(collect(row))); winparams.params...)
    # interval_diff = length(n_intervals) - length(row_intervals)
    
    # calculate reduced values for this row
    reduced_data = [
        vcat([modalreduce(col[r]) for r in intervals] for col in row)
    ]
    result_matrix[row_idx, :] = reduced_data
end



_X = DataFrame(
    [v => [
        [modalreduce(ts[interval]) for interval in intervals]
        for ts in X[!, v]
    ]
    for v in vnames]...
)

col_names = vnames

n_rows = size(X, 1)
n_cols = length(col_names)
result_matrix = Matrix(undef, n_rows, n_cols)

modalreduce === nothing && (modalreduce = mean)

for (row_idx, row) in enumerate(eachrow(X))
    # row_intervals = winparams.type(maximum(length.(collect(row))); winparams.params...)
    # interval_diff = length(n_intervals) - length(intervals)
    
    # calculate reduced values for this row
    reduced_data = [
        vcat([modalreduce(col[r]) for r in intervals] for col in row)]
    result_matrix[row_idx, :] = reduced_data
end

#####################################################

# Current version with many brackets:
@btime begin
    _X = DataFrame(
        [v => [
            [modalreduce(ts[interval]) for interval in intervals]
            for ts in X[!, v]
        ]
        for v in vnames]...
    )
end
# 13.244 ms (164578 allocations: 6.61 MiB)
    a=      DataFrame(
            [v                          => modalreduce(ts[interval]) 
            for interval in intervals 
            for ts in X[!, v] ]
            for v in vnames
            
            
            
        )