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
#                                   references                                 #
# ---------------------------------------------------------------------------- #
# https://github.com/JuliaCollections/OrderedCollections.jl
# https://github.com/brenhinkeller/StaticTools.jl
# https://medium.com/chifi-media/5-simple-ways-to-reduce-memory-usage-in-julia-19bccea6d21b

# ---------------------------------------------------------------------------- #
#                                 prepare dataset                              #
# ---------------------------------------------------------------------------- #
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