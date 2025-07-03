using Test
using MLJ, SoleXplorer
using DataFrames, Random
using SoleData
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# Xr, yr = @load_boston
# Xr = DataFrame(Xr)

Xts, yts = SoleData.load_arff_dataset("NATOPS")

_, ds = prepare_dataset(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
)

@btime begin
    _, ds = prepare_dataset(
        Xc, yc;
        model=(;type=:decisiontree),
        resample = (type=Holdout, params=(;shuffle=true)),
        preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    )
end
# 18.341 μs (308 allocations: 21.27 KiB)
# 17.840 μs (309 allocations: 16.65 KiB)

# ---------------------------------------------------------------------------- #
#                                   references                                 #
# ---------------------------------------------------------------------------- #
# https://github.com/JuliaCollections/OrderedCollections.jl
# https://github.com/brenhinkeller/StaticTools.jl
# https://medium.com/chifi-media/5-simple-ways-to-reduce-memory-usage-in-julia-19bccea6d21b

# ---------------------------------------------------------------------------- #
#                                 prepare dataset                              #
# ---------------------------------------------------------------------------- #
@btime begin
    a = Matrix(Xc)
end
# 1.691 μs (5 allocations: 4.85 KiB)
@btime begin
    a = @views Matrix(Xc)
end
# 1.665 μs (5 allocations: 4.85 KiB)
@btime begin
    a = Matrix(@views Xc)
end
# 1.668 μs (5 allocations: 4.85 KiB)
@btime begin
    c(q::AbstractDataFrame) = Matrix(q)
    a = c(Xc)
end
# 1.5001 μs (5 allocations: 4.85 KiB)
@btime begin
    c(q::Matrix) = q
    a = c(Matrix(Xc))
end
# 1.879 μs (5 allocations: 4.85 KiB)
@btime begin
    c(q::Matrix) = q
    b = Matrix(Xc)
    a = c(b)
end

_, ds = prepare_dataset(
    Xc, yc;
    model=(;type=:xgboost),
    resample = (type=Holdout, params=(;shuffle=true)),
    preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
)
@btime begin
    _, ds = prepare_dataset(
        Xc, yc;
        model=(;type=:xgboost),
        resample = (type=Holdout, params=(;shuffle=true)),
        preprocess=(;train_ratio=0.7, rng=Xoshiro(1)),
    )
end
# 22.526 μs (155 allocations: 15.47 KiB)

using CategoricalArrays

a = Tuple(Symbol.(yts))
@btime a = Tuple(Symbol.(yts))
# 24.596 μs (5 allocations: 5.79 KiB)
sizeof(yts)          # 16
sizeof(Symbol.(yts)) # 2880
sizeof(a)            # 2880

a = Tuple(yr)
@btime a = Tuple(yr)
# 21.932 μs (510 allocations: 16.04 KiB)
sizeof(yr) # 4048
sizeof(a)  # 4048

abstract type AbstractDataset end

struct D11{T,S} <: AbstractDataset
    X           :: Matrix{<:T}
    y           :: AbstractVector{<:S}
    # tt          :: Vector{<:TT_indexes}
    # info        :: DatasetInfo
end

struct D2{T<:AbstractMatrix,S} <: AbstractDataset
    X           :: T
    y           :: S
end

struct D3 <: AbstractDataset
    X           :: Matrix
    y           :: AbstractVector
    # tt          :: Vector{<:TT_indexes}
    # info        :: DatasetInfo
end

a = D11{eltype(ds.X), eltype(ds.y)}(ds.X, ds.y)
b = D2{typeof(ds.X), typeof(ds.y)}(ds.X, ds.y)
c = D3(ds.X, ds.y)
@btime a = D11{eltype(ds.X), eltype(ds.y)}(ds.X, ds.y)
# 369.317 ns (5 allocations: 352 bytes)
@btime b = D2{typeof(ds.X), typeof(ds.y)}(ds.X, ds.y)
# 343.858 ns (5 allocations: 352 bytes)
@btime c = D3(ds.X, ds.y)
# 366.312 ns (3 allocations: 192 bytes)
sizeof(a) # 16
sizeof(b) # 16
sizeof(c)

@btime begin
    test = []
    for i in 1:length(a.y)
        if eltype(a.X[i]) == Float64
            test = vcat(test, a.X[i])
        end
    end
end
@btime begin
    test = []
    for i in 1:length(b.y)
        if eltype(b.X[i]) == Float64
            test = vcat(test, b.X[i])
        end
    end
end
@btime begin
    test = []
    for i in 1:length(c.y)
        if eltype(c.X[i]) == Float64
            test = vcat(test, c.X[i])
        end
    end
end
# 56.404 μs (902 allocations: 112.03 KiB)

# ---------------------------------------------------------------------------- #
#                             optimized versions                               #
# ---------------------------------------------------------------------------- #
@btime begin
    test = typeof(a.X)[]
    for i in 1:length(a.y)
        if eltype(a.X[i]) == Float64
            test = vcat(test, a.X[i])
        end
    end
end
# 56.490 μs (902 allocations: 112.03 KiB)
@btime begin
    test = typeof(a.X)[]
    for i in 1:length(a.y)
        if eltype(a.X[i]) == Float64
            test = vcat(test, view(a.X, i, :))
        end
    end
end
# 320.850 μs (4440 allocations: 464.11 KiB)
@btime begin
    test = typeof(a.X)[]
    for i in 1:length(a.y)
        if eltype(a.X[i]) == Float64
            test = vcat(test, @views a.X[i])
        end
    end
end
# 57.164 μs (902 allocations: 112.03 KiB)

# Version 2: Even more efficient - collect all at once
@btime begin
    if eltype(a.X) == Float64
        test = vec(a.X)  # Simply vectorize the entire matrix
    end
end
# 168.723 ns (1 allocation: 32 bytes)
@btime begin
    if eltype(a.X) == Float64
        test = @views vec(a.X)  # Simply vectorize the entire matrix
    end
end
# 167.834 ns (1 allocation: 32 bytes)
@btime begin
    if eltype(a.X) == Float64
        test = [a.X[i, j] for i in 1:size(a.X, 1), j in 1:size(a.X, 2)]
        # test = vec(test)  # Flatten if needed
    end
end
# 31.690 μs (615 allocations: 14.63 KiB)

@btime a(ds)
# 495.175 ns (5 allocations: 144 bytes)
@btime view(ds.tt[1].train, 1:10)
# 93.463 ns (2 allocations: 80 bytes)
@btime begin
    @views ds.tt[1].train[1:10]
end
# 94.273 ns (2 allocations: 80 bytes)
@btime begin
    @view ds.tt[1].train[1:10]
end
# 95.266 ns (2 allocations: 80 bytes)
@btime view(ds.tt[1].train, :)
# 617.349 ns (14 allocations: 272 bytes)
@btime begin
    @views ds.tt[1].train
end
# 74.085 ns (1 allocation: 32 bytes)
@btime begin
    @view ds.tt[1].train[:]
end
# 613.644 ns (14 allocations: 272 bytes)
@btime a = [i.train for i in ds.tt]
# 289.963 ns (3 allocations: 80 bytes)
@btime a = [@views i.train for i in ds.tt]
# 305.072 ns (3 allocations: 80 bytes)
@btime a= [i.train for i in @views ds.tt]
# 288.827 ns (3 allocations: 80 bytes)
@btime a = [view(i.train, :) for i in ds.tt]
# 293.299 ns (3 allocations: 112 bytes)

# Using map - most direct replacement
@btime map(i -> i.train, ds.tt)
# 490.206 ns (3 allocations: 80 bytes)

# Using map with getproperty for even cleaner syntax
@btime map(x -> getproperty(x, :train), ds.tt)
# 455.706 ns (3 allocations: 80 bytes)

# Using broadcasting (often fastest)
@btime a = getproperty.(ds.tt, :train)
# 100.827 ns (4 allocations: 112 bytes)

@btime a = [getproperty(i, :train) for i in ds.tt]
# 100.827 ns (4 allocations: 112 bytes)

# If you want to flatten all train indices into a single vector using reduce
@btime reduce(vcat, map(i -> i.train, ds.tt))

# Or combine map and reduce in one step
@btime reduce(vcat, (i.train for i in ds.tt))

# Using mapreduce if you want to apply a function and reduce in one step
@btime mapreduce(i -> i.train, vcat, ds.tt)

# If you want unique indices across all train sets
@btime reduce(union, map(i -> i.train, ds.tt))

# Using collect with generator (similar to comprehension but more explicit)
@btime collect(i.train for i in ds.tt)
# 290.639 ns (3 allocations: 80 bytes)

# Broadcasting with anonymous function
@btime a = (x -> x.train).(ds.tt)
# 85.765 ns (3 allocations: 80 bytes)

# Broadcasting with pipe operator
@btime a = ds.tt .|> x -> x.train
# 1.566 μs (8 allocations: 280 bytes)

@btime a= ds.tt[1].train
# 72.543 ns (1 allocation: 32 bytes)
@btime a = @views ds.tt[1].train
# 69.109 ns (1 allocation: 32 bytes)

a1(ds :: Dataset) :: Vector{Vector{<:Integer}} = (x -> x.train).(ds.tt)
# a2(ds :: Dataset) :: Vector{Vector{<:Integer}} = (x -> get_train(x)).(ds.tt)
a3(ds :: Dataset) :: Vector{Vector{<:Integer}} = [x.train for x in ds.tt]
a4(ds :: Dataset) = collect(x.train for x in ds.tt)
# Using varargs - returns multiple vectors as separate arguments


get_train(ds :: Dataset, i :: Integer) :: Vector{<:Integer} = ds.tt[i].train

@btime a1(ds)
# 1.806 μs (14 allocations: 480 bytes)
@btime a3(ds)
# 476.585 ns (5 allocations: 144 bytes)
@btime a4(ds)
# 294.274 ns (3 allocations: 80 bytes)
@btime a5(ds)

get_X(ds::Dataset) :: Matrix{T} where T     = ds.X

get_X(ds)

@btime begin
    strings = ["string_$(i)_$(rand(1000:9999))" for i in 1:100]
    b = @views strings[1:30]
end
# 27.274 μs (802 allocations: 25.91 KiB)

@btime begin
    symbols = [Symbol("symbol_$(i)_$(rand(1000:9999))") for i in 1:100]
    b = @views symbols[1:30]
end
# 210.282 μs (802 allocations: 25.91 KiB)

strings = ["string_$(i)_$(rand(1000:9999))" for i in 1:100]
symbols = [Symbol("symbol_$(i)_$(rand(1000:9999))") for i in 1:100]
@btime Symbol.($strings)
# 11.686 μs (2 allocations: 928 bytes)
@btime map(Symbol, $strings)
# 11.903 μs (2 allocations: 928 bytes)
@btime [Symbol(x) for x in $strings]
# 11.864 μs (2 allocations: 928 bytes)

@btime String.($symbols)
# 1.617 μs (102 allocations: 4.03 KiB)
@btime map(String, $symbols)
# 1.623 μs (102 allocations: 4.03 KiB)
@btime [String(x) for x in $symbols]
# 1.661 μs (102 allocations: 4.03 KiB)
@btime string.($symbols)
# 1.649 μs (102 allocations: 4.03 KiB)

# Memory-efficient streaming approach (for huge datasets)
symbols_generator(strings::AbstractVector{<:AbstractString})::Base.Generator{Vector{String}} = (Symbol(s) for s in strings)
strings_generator(symbols) = (String(s) for s in symbols)
symbols_generator(strings::MLJ.CategoricalArray) = (Symbol(s) for s in strings)

a = symbols_generator(yc)

@btime collect(symbols_generator($strings))
# 7.298 μs (2 allocations: 928 bytes)
@btime collect(strings_generator($symbols))

using CategoricalArrays
categorical_data = categorical(["category_$(rand(1:10))" for i in 1:100])

@btime collect(symbols_generator($categorical_data))

@btime String.($symbols)
@btime Symbol.($strings)

@btime begin
    for col in eachcol(Xc)
        # skip cols with only scalar values
        any(el -> el isa AbstractArray, col) || continue
        
        # find first array element to use as reference
        ref_idx = findfirst(el -> el isa AbstractArray, col)
        ref_idx === nothing && continue
        
        ref_size = size(col[ref_idx])
        
        # check if any array element has different size (short-circuit)
        if any(col) do el
                el isa AbstractArray && size(el) != ref_size
            end
            return false
        end
    end
    return true
end
# row
# 130.744 μs (2401 allocations: 65.64 KiB)
# col
# 949.250 ns (5 allocations: 144 bytes)

@btime begin
    for row in eachrow(Xc)
        a = row
    end
    return true
end
# 8.575 μs (301 allocations: 14.08 KiB)

# ---------------------------------------------------------------------------- #
#                          Memory-optimized DataFrame iteration                #
# ---------------------------------------------------------------------------- #
@btime begin
    for i in 1:nrow(Xc)
        a = @view Xc[i, :]
    end
    return true
end
# 12.874 μs (301 allocations: 11.75 KiB)

@btime begin
    @views for row in eachrow(Xc)
        a = row
    end
    return true
end
# 8.608 μs (301 allocations: 14.08 KiB)

@btime begin
    col_views = [@view Xc[:, j] for j in 1:ncol(Xc)]
    for i in 1:nrow(Xc)
        # Access as col_views[j][i] instead of Xc[i, j]
        # This is very efficient for column-wise operations
    end
    return true
end
# 12.595 μs (213 allocations: 6.11 KiB)

# Option 12: Memory-efficient column iteration with views
@btime begin
    for col_name in names(Xc)
        col_view = @view Xc[:, col_name]
        # Process entire column at once
    end
    return true
end
# 4.764 μs (62 allocations: 1.28 KiB)

function check_row_consistency(X::DataFrame) 
    ref_length = length.(X[!, 1])

    for col in eachcol(X)
        all(length.(col) == ref_length) || throw(ArgumentError("Elements within each row must have consistent dimensions"))
    end
end

@btime check_row_consistency(Xc)
# 386.926 ns (3 allocations: 1.27 KiB)

@btime check_row_consistency(Xts)
# 443.713 ns (4 allocations: 2.90 KiB)

@btime begin
    @eachrow! ds.X begin
        any(el -> el isa AbstractArray, names(ds.X))
    end
end
# 129.563 μs (2401 allocations: 65.64 KiB)
# 44.684 μs (930 allocations: 39.08 KiB)

function check_row_consistency(X::AbstractDataFrame) 
    for row in eachrow(X)
        # skip cols with only scalar values
        all(eltype.(row)) <: AbstractArray || continue
        
        # # find first array element to use as reference
        # ref_idx = findfirst(el -> el isa AbstractArray, col)
        # ref_idx === nothing && continue
        
        # ref_size = size(col[ref_idx])
        
        # # check if any array element has different size (short-circuit)
        # if any(col) do el
        #         el isa AbstractArray && size(el) != ref_size
        #     end
        #     return false
        # end
    end
    return true
end

# Even more optimized version with @inbounds
function check_row_consistency_fast(X::AbstractDataFrame) 
    for col in eachcol(X)
        # Check eltype first
        eltype(col) <: AbstractArray || continue
        
        isempty(col) && continue
        
        # Get reference size from first element
        @inbounds ref_size = size(col[1])
        
        # Check remaining elements with bounds checking disabled
        @inbounds for i in 2:length(col)
            size(col[i]) != ref_size && return false
        end
    end
    return true
end

# Type-stable version that avoids dynamic dispatch
function check_row_consistency_typed(X::AbstractDataFrame) 
    for col in eachcol(X)
        _check_column_consistency(col)
    end
    return true
end

@inline function _check_column_consistency(col::AbstractVector{T}) where T
    T <: AbstractArray || return true
    
    isempty(col) && return true
    
    @inbounds ref_size = size(col[1])
    @inbounds for i in 2:length(col)
        size(col[i]) != ref_size && return false
    end
    return true
end

@btime check_row_consistency(Xc)
# 545.723 ns (0 allocations: 0 bytes)

@btime check_row_consistency(Xts)
# 24.580 μs (48 allocations: 768 bytes)



