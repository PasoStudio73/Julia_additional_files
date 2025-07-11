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
        Xc, yc;
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

resample=(type=Holdout(shuffle=true), train_ratio=0.7, valid_ratio=0.1, rng=Xoshiro(1))
a=SX.partition(yc; resample...)

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


# define column names and prepare data structure based on treatment type
if treat == :aggregate        # propositional
    if n_intervals == 1
        # Apply feature to whole time series
        _X = DataFrame(
            [Symbol(f, "(", v, ")") => [f(ts) for ts in X[!, v]]
                for f in features
                for v in vnames]...)
    else
        # apply feature to specific intervals
        _X = DataFrame(
            [Symbol(f, "(", v, ")w", i) => f.(getindex.(X[!, v], Ref(interval)))
            for f in features
            for v in vnames
            for (i, interval) in enumerate(intervals)]...)
    end

elseif treat == :reducesize   # modal
    @btime begin
    _X = DataFrame(
        [v => [modalreduce.(X[!, v][interval]) for interval in intervals]
                # for row in 1:nrows(X)
        # [[modalreduce(@views ts[interval]) for interval in intervals]
        #     for ts in X[!, v]
        # ] 
        for v in vnames]
    )
            end

            @btime begin
    # Pre-allocate everything upfront
    n_rows = nrows(X)
    n_vars = length(vnames)
    n_intervals = length(intervals)
    
    # Single allocation for all the result vectors
    all_vectors = Matrix{Vector{Float64}}(undef, n_rows, n_vars)
    
    # Fill the matrix efficiently
    @inbounds for (var_idx, v) in enumerate(vnames)
        col_data = X[!, v]  # Get column once
        
        for row_idx in 1:n_rows
            ts = col_data[row_idx]
            # Pre-allocate the result vector for this cell
            result_vec = Vector{Float64}(undef, n_intervals)
            
            # Fill the vector with reduced values
            for (int_idx, interval) in enumerate(intervals)
                result_vec[int_idx] = modalreduce(ts[interval])
            end
            
            all_vectors[row_idx, var_idx] = result_vec
        end
    end
    
    # Create DataFrame with pre-computed data
    pairs = Vector{Pair{Symbol, Vector{Vector{Float64}}}}(undef, n_vars)
    @inbounds for (var_idx, v) in enumerate(vnames)
        pairs[var_idx] = v => all_vectors[:, var_idx]
    end
    
    _X = DataFrame(pairs...)
end

# Alternative: Even more efficient with single large allocation
@btime begin
    n_rows = nrows(X)
    n_vars = length(vnames)
    n_intervals = length(intervals)
    
    # Single flat array to hold all data
    flat_data = Vector{Float64}(undef, n_rows * n_vars * n_intervals)
    
    # Fill the flat array
    @inbounds for (var_idx, v) in enumerate(vnames)
        col_data = X[!, v]
        base_idx = (var_idx - 1) * n_rows * n_intervals
        
        for row_idx in 1:n_rows
            ts = col_data[row_idx]
            row_base = base_idx + (row_idx - 1) * n_intervals
            
            for (int_idx, interval) in enumerate(intervals)
                flat_data[row_base + int_idx] = modalreduce(ts[interval])
            end
        end
    end
    
    # Reshape into the desired structure
    pairs = Vector{Pair{Symbol, Vector{Vector{Float64}}}}(undef, n_vars)
    @inbounds for (var_idx, v) in enumerate(vnames)
        col_vectors = Vector{Vector{Float64}}(undef, n_rows)
        base_idx = (var_idx - 1) * n_rows * n_intervals
        
        for row_idx in 1:n_rows
            start_idx = base_idx + (row_idx - 1) * n_intervals + 1
            end_idx = start_idx + n_intervals - 1
            col_vectors[row_idx] = flat_data[start_idx:end_idx]
        end
        
        pairs[var_idx] = v => col_vectors
    end
    
    _X = DataFrame(pairs...)
end

    _X = DataFrame(
        [v => modalreduce.(hcat(X[! ,v][i] for i in intervals)...)
        for v in vnames]...)

     _X = [modalreduce(r[i]) for c in eachcol(X) for r in c for i in intervals]

    for v in vnames
        modalreduce.(hcat(X[! ,v][i] for i in intervals)...)
     end
     modalreduce.(hcat.(X[! ,v][i] for i in intervals)...)
     

@btime begin
    # Build DataFrame column by column without pre-allocation
    _X = DataFrame()
    
    for v in vnames
        col_data = X[!, v]  # Get column reference once
        
        # Use map with views - no intermediate allocations
        reduced_column = map(col_data) do ts
            # Use views to avoid copying data, map to apply modalreduce
            map(intervals) do interval
                modalreduce(@view ts[interval])
            end
        end
        
        # Add column directly to DataFrame
        _X[!, v] = reduced_column
    end
end

# Alternative: Even more efficient with mapreduce-style approach
@btime begin
    _X = DataFrame()
    
    # Process each variable column
    for v in vnames
        col_data = @view X[!, v]  # Use view for the entire column
        
        # Single map operation - most efficient
        _X[!, v] = map(col_data) do ts
            # Map over intervals using views - no copying
            [modalreduce(@view ts[interval]) for interval in intervals]
        end
    end
end

# Ultra-efficient: Use reduce pattern for minimal allocations
@btime begin
    # Start with empty DataFrame
    _X = DataFrame()
    
    # Use fold/reduce pattern to build columns efficiently
    for v in vnames
        # Single allocation per column using efficient iterator
        _X[!, v] = [
            # Use views and direct iteration - minimal overhead
            [modalreduce(@view X[row, v][interval]) for interval in intervals]
            for row in 1:nrows(X)
        ]
    end
end

@btime begin
    # Start with empty DataFrame

    
    # Use fold/reduce pattern to build columns efficiently
    _X = DataFrame(begin
        for v in vnames
            # Single allocation per column using efficient iterator
            [
                # Use views and direct iteration - minimal overhead
                [modalreduce(@view X[row, v][interval]) for interval in intervals]
                for row in 1:nrows(X)
            ]
        end
    end)
end

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

abstract type AbstractDataSet end
abstract type AbstractPropositional end
abstract type AbstractModal end

mutable struct PropositionalDataSet <: AbstractDataSet
    mach    :: MLJ.Machine
    ttpairs :: PartitionIdxs
    pinfo :: PartitionInfo
end

mutable struct ModalDataSet <: AbstractDataSet
    mach    :: MLJ.Machine
    ttpairs :: PartitionIdxs
    pinfo :: PartitionInfo
    tinfo :: TreatmentInfo
end

function DataSet(
    mach::MLJ.Machine,
    ttpairs::PartitionIdxs,
    pinfo::PartitionInfo;
    tinfo::Union{TreatmentInfo, Nothing} = nothing
)
    isnothing(tinfo) ?
        PropositionalDataSet(mach, ttpairs, pinfo) :
        ModalDataSet(mach, ttpairs, pinfo, tinfo)
end

# Method 1: Using map with view (most efficient - no copying)
@btime first_10 = map(v -> @view(v[1:10]), X[!, 1])
# 2.850 μs (4 allocations: 14.15 KiB)

# Method 2: Using broadcasting with getindex (also efficient)
@btime first_10 = getindex.(X[!, 1], Ref(1:10))
# 11.300 μs (725 allocations: 53.57 KiB)

# Method 3: Using map with copying (less efficient but sometimes needed)
@btime first_10 = map(v -> v[1:10], X[!, 1])

# Method 4: If you want to modify X in place to keep only first 10 elements
@btime X[!, 1] = map(v -> v[1:10], X[!, 1])

# Method 5: For all columns at once
@btime _X = DataFrame([
    col => map(v -> modalreduce.(@views v[i] for i in intervals), X[!, col])
    for col in names(X)
])
# 137.930 μs (249 allocations: 684.48 KiB)

# Method 6: Most efficient for all columns - modify in place
@btime for col in names(X)
    X[!, col] = map(v -> v[1:10], X[!, col])
end

        _X = DataFrame(
            [v => [[modalreduce(ts[interval]) for interval in intervals]
                for ts in X[!, v]]
            for v in vnames]...
        )
@btime begin
                    _X = DataFrame(
                Symbol(f, "(", v, ")w", i) => f.(getindex.(X[!, v], Ref(interval)))
                for f in features
                for v in vnames
                for (i, interval) in enumerate(intervals)
            )
                end

                @btime begin
                        _X = DataFrame(
                Symbol(f, "(", v, ")w", i) => f.(map(v -> @view(v[interval]), X[!, v]))
                for f in features
                for v in vnames
                for (i, interval) in enumerate(intervals)
            )
                end

            map(v -> @view(v[interval]), X[!, v])

            getindex.(X[!, 1], Ref(1:10))