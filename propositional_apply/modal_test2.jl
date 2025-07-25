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

a = setup_dataset(
        Xc, yc;
        model=DecisionTreeClassifier(),
        win=AdaptiveWindow(nwindows=3, relative_overlap=0.1),
        resample=(type=Holdout(shuffle=true), train_ratio=0.7, rng=Xoshiro(1))
)

@btime begin
    a = setup_dataset(
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

@btime begin
    _X = DataFrame(
        (Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
            for f in features
            for v in vnames)...)
end
# 4.463 ms (901 allocations: 310.82 KiB)

@btime begin
    pairs = mapreduce(vcat, features) do f
        map(vnames) do v
            Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
        end
    end
    _X = DataFrame(pairs...)
end
# 4.489 ms (787 allocations: 307.20 KiB)

@btime begin
    _X = DataFrame()
    for f in features
        @simd for v in vnames
            col_name = Symbol(f, "(", v, ")")
            _X[!, col_name] = collect(f(ts) for ts in X[!, v])
        end
    end
end
# 4.435 ms (557 allocations: 167.55 KiB)

# Ultimate optimization: Custom broadcast function
@inline function apply_features_vectorized!(
    X::DataFrame,
    X_col::Vector{Vector{Float64}},
    feature_func::Function,
    col_name::Symbol
)
    X[!, col_name] = @inbounds [feature_func(X_col[i]) for i in 1:length(X_col)]
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

@btime begin
    n_rows = length(X[!, first(vnames)])
    n_cols = length(features) * length(vnames)
    
    # Single allocation
    data_matrix = Matrix{Float64}(undef, n_rows, n_cols)
    col_names = Vector{Symbol}(undef, n_cols)
    
    col_idx = 1
    @inbounds for f in features
        for v in vnames
            col_names[col_idx] = Symbol(f, "(", v, ")")
            apply_features_vectorized!(data_matrix, X[!, v], f, col_idx)
            col_idx += 1
        end
    end
    
    _X = DataFrame(data_matrix, col_names)
end
# 4.425 ms (559 allocations: 300.94 KiB)

#############################################################################
# Method 1: Using map with flatten
@btime begin
    pairs = map(features) do f
        map(vnames) do v
            Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
        end
    end |> Iterators.flatten
    _X = DataFrame(pairs...)
end
# 4.483 ms (894 allocations: 311.54 KiB)

# Method 2: Using mapreduce with vcat
@btime begin
    pairs = mapreduce(vcat, features) do f
        map(vnames) do v
            Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
        end
    end
    _X = DataFrame(pairs...)
end
# 4.489 ms (787 allocations: 307.20 KiB)

# Method 3: Single map over Iterators.product (most efficient)
@btime begin
    pairs = map(Iterators.product(features, vnames)) do (f, v)
        Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
    end |> vec  # Convert matrix to vector
    _X = DataFrame(pairs...)
end
# 4.422 ms (1051 allocations: 312.34 KiB)

# Method 4: Nested map (cleanest)
@btime begin
    pairs = map(f -> map(v -> Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v]), vnames), features) |> Iterators.flatten
    _X = DataFrame(pairs...)
end
# 4.488 ms (894 allocations: 311.54 KiB)

# Method 5: Using reduce for ultimate efficiency
@btime begin
    pairs = reduce(vcat, 
        map(f -> map(v -> Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v]), vnames), features)
    )
    _X = DataFrame(pairs...)
end
# 4.483 ms (799 allocations: 307.57 KiB)

###############################################################################
# Method 1: Direct DataFrame construction with named tuples (most efficient)
@btime begin
    _X = DataFrame(Dict(
        Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
        for f in features
        for v in vnames
    ))
end
# 4.500 ms (940 allocations: 321.16 KiB)

# Method 2: Using NamedTuple (very efficient)
@btime begin
    nt = NamedTuple(
        Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
        for f in features
        for v in vnames
    )
    _X = DataFrame(nt)
end
# 4.456 ms (911 allocations: 313.15 KiB)

# Method 3: Build DataFrame incrementally (avoiding splat entirely)
@btime begin
    _X = DataFrame()
    for f in features
        for v in vnames
            col_name = Symbol(f, "(", v, ")")
            _X[!, col_name] = collect(f(ts) for ts in X[!, v])
        end
    end
end
# 4.447 ms (603 allocations: 168.98 KiB)

# Method 4: Using OrderedDict for guaranteed column order
using OrderedCollections
@btime begin
    _X = DataFrame(OrderedDict(
        Symbol(f, "(", v, ")") => collect(f(ts) for ts in X[!, v])
        for f in features
        for v in vnames
    ))
end
# 4.459 ms (913 allocations: 312.18 KiB)

# Method 5: Pre-allocate column names and data separately
@btime begin
    col_names = [Symbol(f, "(", v, ")") for f in features for v in vnames]
    col_data = [collect(f(ts) for ts in X[!, v]) for f in features for v in vnames]
    _X = DataFrame(col_data, col_names)
end
# 4.426 ms (954 allocations: 311.10 KiB)

# Method 6: Most efficient - direct matrix construction
@btime begin
    n_rows = length(X[!, first(vnames)])
    n_cols = length(features) * length(vnames)
    
    # Pre-allocate matrix
    data_matrix = Matrix{Float64}(undef, n_rows, n_cols)
    col_names = Vector{Symbol}(undef, n_cols)
    
    col_idx = 1
    for f in features
        for v in vnames
            col_names[col_idx] = Symbol(f, "(", v, ")")
            data_matrix[:, col_idx] = collect(f(ts) for ts in X[!, v])
            col_idx += 1
        end
    end
    
    _X = DataFrame(data_matrix, col_names)
end
# 4.509 ms (799 allocations: 440.81 KiB)

######################################################################################

# Ultimate optimization: Custom broadcast function
@inline function apply_features_vectorized!(result::Matrix{Float64}, X_col::Vector{Vector{Float64}}, feature_func::Function, col_idx::Int)
    @inbounds @simd for i in eachindex(X_col)
        result[i, col_idx] = feature_func(X_col[i])
    end
end

@btime begin
    n_rows = length(X[!, first(vnames)])
    n_cols = length(features) * length(vnames)
    
    # Single allocation
    data_matrix = Matrix{Float64}(undef, n_rows, n_cols)
    col_names = Vector{Symbol}(undef, n_cols)
    
    col_idx = 1
    @inbounds for f in features
        for v in vnames
            col_names[col_idx] = Symbol(f, "(", v, ")")
            apply_features_vectorized!(data_matrix, X[!, v], f, col_idx)
            col_idx += 1
        end
    end
    
    _X = DataFrame(data_matrix, col_names)
end
# 4.425 ms (559 allocations: 300.94 KiB)

# Memory-pool optimization for extreme performance
@btime begin
    # Pre-allocate all memory upfront
    n_rows = length(X[!, first(vnames)])
    n_cols = length(features) * length(vnames)
    
    # Use unsafe operations for maximum speed
    data_ptr = Libc.malloc(sizeof(Float64) * n_rows * n_cols)
    data_matrix = unsafe_wrap(Matrix{Float64}, Ptr{Float64}(data_ptr), (n_rows, n_cols))
    
    col_names = Vector{Symbol}(undef, n_cols)
    
    try
        col_idx = 1
        @inbounds @fastmath for f in features
            for v in vnames
                col_names[col_idx] = Symbol(f, "(", v, ")")
                col_data = X[!, v]
                
                # Manual loop unrolling for small datasets
                @simd for i in 1:n_rows
                    data_matrix[i, col_idx] = f(col_data[i])
                end
                
                col_idx += 1
            end
        end
        
        _X = DataFrame(data_matrix, col_names; copycols=false)
    finally
        Libc.free(data_ptr)
    end
end

# Final ultra-optimized version combining all techniques
@btime begin
    @inbounds @fastmath  begin
        n_rows = length(X[!, first(vnames)])
        n_cols = length(features) * length(vnames)
        
        # Stack allocation for small arrays
        data_matrix = Matrix{Float64}(undef, n_rows, n_cols)
        col_names = Vector{Symbol}(undef, n_cols)
        
        # Unroll loops for known small sizes
        col_idx = 1
        for f in features
            for v in vnames
                col_names[col_idx] = Symbol(f, "(", v, ")")
                
                # Vectorized assignment
                col_data = X[!, v]
                for i in 1:n_rows
                    data_matrix[i, col_idx] = f(col_data[i])
                end
                
                col_idx += 1
            end
        end
        
        DataFrame(data_matrix, col_names; copycols=false)
    end
end

# Parallel processing version for large datasets
using Base.Threads

@btime begin
    n_rows = length(X[!, first(vnames)])
    n_cols = length(features) * length(vnames)
    
    data_matrix = Matrix{Float64}(undef, n_rows, n_cols)
    col_names = Vector{Symbol}(undef, n_cols)
    
    # Parallel computation
    @threads for feature_idx in 1:length(features)
        f = features[feature_idx]
        for var_idx in 1:length(vnames)
            v = vnames[var_idx]
            col_idx = (feature_idx - 1) * length(vnames) + var_idx
            
            col_names[col_idx] = Symbol(f, "(", v, ")")
            col_data = X[!, v]
            
            @inbounds @simd for i in 1:n_rows
                data_matrix[i, col_idx] = f(col_data[i])
            end
        end
    end
    
    _X = DataFrame(data_matrix, col_names; copycols=false)
end

#######################################################################
using Base.Threads

# Helper for tight loop kernel - forces inlining and vectorization
@inline function compute_feature_for_column!(
    flat_buffer::Vector{Float64}, 
    feature::Function, 
    column::Vector{Vector{Float64}}, 
    base_offset::Int
)
    @inbounds @simd for i in eachindex(column)
        flat_buffer[base_offset + i] = feature(column[i])
    end
    nothing
end

# Main optimized function
@btime begin
    # Pre-compute all dimensions
    n_rows = length(X[!, first(vnames)])
    n_features = length(features)
    n_vars = length(vnames)
    n_cols = n_features * n_vars
    
    # Pre-allocate a SINGLE flat buffer for all data - minimizes allocations
    flat_buffer = Vector{Float64}(undef, n_rows * n_cols)
    col_names = Vector{Symbol}(undef, n_cols)
    
    # Set up thread-local storage for indices to avoid race conditions
    n_threads = Threads.nthreads()
    
    # Process each feature/variable combination in parallel
    # Using flattened 1D iteration for better load balancing
    @threads for idx in 1:n_cols
        # Convert linear index to feature/variable indices
        f_idx = (idx - 1) ÷ n_vars + 1
        v_idx = (idx - 1) % n_vars + 1
        
        f = features[f_idx]
        v = vnames[v_idx]
        
        # Pre-compute base offset for this column in flat buffer
        base = (idx - 1) * n_rows
        
        # Get column data once
        col_data = X[!, v]
        
        # Set column name
        col_names[idx] = Symbol(f, "(", v, ")")
        
        # Compute feature values - tightest possible kernel with all optimizations
        compute_feature_for_column!(flat_buffer, f, col_data, base)
    end
    
    # Reshape buffer to matrix WITHOUT copying (zero-cost reshape)
    data_matrix = reshape(flat_buffer, n_rows, n_cols)
    
    # Create DataFrame at the very end - minimizes DataFrame overhead
    # copycols=false avoids a final copy
    _X = DataFrame(data_matrix, col_names; copycols=false)
end