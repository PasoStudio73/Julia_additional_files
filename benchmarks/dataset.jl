using SoleXplorer
using BenchmarkTools
using Random

struct DS1{T<:SoleXplorer.AbstractMatrix,S} <: SoleXplorer.AbstractDataset
    X           :: T
    y           :: S
    tt          :: Union{SoleXplorer.TT_indexes, AbstractVector{<:SoleXplorer.TT_indexes}}
    Xtrain      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xvalid      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xtest       :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    yvalid      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function DS1(X::T, y::S, tt, info) where {T<:SoleXplorer.AbstractMatrix,S}
        Xtrain = @views X[tt.train, :]
        Xvalid = @views X[tt.valid, :]
        Xtest  = @views X[tt.test,  :]
        ytrain = @views y[tt.train]
        yvalid = @views y[tt.valid]
        ytest  = @views y[tt.test]
        new{T,S}(X, y, tt, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    end
end

struct DS2{T<:SoleXplorer.AbstractMatrix,S} <: SoleXplorer.AbstractDataset
    Xtrain      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xvalid      :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    Xtest       :: Union{AbstractMatrix, Vector{<:AbstractMatrix}}
    ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    yvalid      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
    ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

    function DS2(X::T, y::S, tt, info) where {T<:SoleXplorer.AbstractMatrix,S}
        Xtrain = @views X[tt.train, :]
        Xvalid = @views X[tt.valid, :]
        Xtest  = @views X[tt.test,  :]
        ytrain = @views y[tt.train]
        yvalid = @views y[tt.valid]
        ytest  = @views y[tt.test]
        new{T,S}(Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
    end
end

###############################################
Xm = rand(Xoshiro(11), 10, 5)
ym = rand(Xoshiro(11), 10)

column_eltypes = eltype.(eachcol(Xm))

ds_info = SoleXplorer.DatasetInfo(
    :regression,
    :aggregate,
    mean,
    0.7,
    0.0,
    Xoshiro(11),
    false,
    string.(Xm[1,:])
)

dstest = SoleXplorer._partition(ym, 0.7, 1.0, nothing, Xoshiro(11))

test1 = DS1(Xm, ym, dstest, ds_info)
test2 = DS2(Xm, ym, dstest, ds_info)

function f2(d)
    @show "PASO"
end

function f1(d)
    f2(d)
end

@btime f1(test1)
# 4.234 μs (9 allocations: 240 bytes)
@btime f1(test2)