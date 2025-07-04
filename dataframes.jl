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

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
a = MLJ.machine(tree, Xc, yc; cache=true)
b = MLJ.machine(tree, Xr, yr; cache=true)
c = MLJ.machine(tree, Xts, yts; cache=true)

@btime a = MLJ.machine(tree, Xc, yc; cache=true)
# 10.735 μs (51 allocations: 1.94 KiB)
@btime a = MLJ.machine(tree, Xc, yc; cache=false)
# 10.686 μs (51 allocations: 1.94 KiB)

e = evaluate(
    tree, Xc, yc;
    resampling=CV(shuffle=true),
    measures=[log_loss, accuracy],
    per_observation=false,
    verbosity=0
)

model, mach, ds = symbolic_analysis(
    Xc, yc;
    model=(;type=:decisiontree),
    resample = (type=CV, params=(;shuffle=true)),
    measures=(log_loss, accuracy)
)

@btime begin
    Tree = @load DecisionTreeClassifier pkg=DecisionTree
    tree = Tree()
    e1f = evaluate(
        tree, Xc, yc;
        resampling=CV(shuffle=true),
        measures=[log_loss, accuracy],
        per_observation=false,
        verbosity=0
    )
end
# 1.734 ms (10593 allocations: 649.28 KiB)

@btime begin
    model, _, _ = symbolic_analysis(
        Xc, yc;
        model=(;type=:decisiontree),
        resample = (type=CV, params=(;shuffle=true)),
        measures=(log_loss, accuracy)
    )
end
# 3.455 ms (33731 allocations: 1.93 MiB)

@btime args = source((Xc, yc))
# 3.989 μs (22 allocations: 832 bytes)

#######################################################################
using  SoleBase: Label, CLabel, RLabel, XGLabel
import MLJ: MLJType

abstract type AbstractSource     <: MLJType        end
abstract type AbstractSoleSource <: AbstractSource end

struct TableSource{T<:DataFrame} <: AbstractSoleSource
    data :: T
end

struct VectorSource{S<:Label, T<:AbstractVector{S}} <: AbstractSoleSource
    data :: T
end

source(X::T) where {T<:DataFrame} = TableSource{T}(X)
source(X::T) where {S, T<:AbstractVector{S}} = VectorSource{S,T}(X)

# source(Xs::Source; args...) = Xs

Base.isempty(X::AbstractSoleSource)  = isempty(X.data)

nrows_at_source(X::TableSource)  = nrows(X.data)
nrows_at_source(X::VectorSource) = length(X.data)

# select rows in a TableSource
# examples:
# ts(rows=1:10)
# ts(rows=:)    # select all rows
function (X::TableSource)(; rows=:)
    rows == (:) && return X.data
    return @views Xc[rows, :]
end
# select elements in a VectorSource
function (X::VectorSource)(; rows=:)
    rows == (:) && return X.data
    return @views X.data[rows]
end

color(::AbstractSoleSource) = :yellow

### test ############################################################
at = source(a.args[1].data)
am = MLJ.source(a.args[1].data)
@btime at = source(a.args[1].data)
# 291.879 ns (1 allocation: 16 bytes)
@btime am = MLJ.source(a.args[1].data)
# 3.849 μs (17 allocations: 640 bytes)
@btime at(rows=1:10)
# 112.297 ns (2 allocations: 80 bytes)
@btime am(rows=1:10)
# 1.589 μs (34 allocations: 1.98 KiB)

at = source(a.args[2].data)
am = MLJ.source(a.args[2].data)
@btime at = source(a.args[2].data)
# 297.239 ns (1 allocation: 16 bytes)
@btime am = MLJ.source(a.args[2].data)
# 519.812 ns (1 allocation: 32 bytes)
@btime at(el=1:10)
# 155.728 ns (10 allocations: 656 bytes)
# con @views: 23.548 ns (1 allocation: 48 bytes)
@btime am(rows=1:10)
# 180.216 ns (11 allocations: 688 bytes)

### implemented
at = SX.source(a.args[1].data)
am = MLJ.source(a.args[1].data)
at = SX.source(a.args[2].data)
am = MLJ.source(a.args[2].data)

at = SX.source(b.args[1].data)
am = MLJ.source(b.args[1].data)
at = SX.source(b.args[2].data)
am = MLJ.source(b.args[2].data)

at = SX.source(c.args[1].data)
am = MLJ.source(c.args[1].data)
at = SX.source(c.args[2].data)
am = MLJ.source(c.args[2].data)
