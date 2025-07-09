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

# model, mach, ds = symbolic_analysis(
a = SX._prepare_dataset(
    Xts, yts;
    model=(;type=modaldecisiontree, params=(;conditions=[maximum])),
    resample = (type=CV, params=(;shuffle=true)),
    measures=(log_loss, accuracy),
    preprocess=(rng=Xoshiro(1), train_ratio=0.7),
)

X, y = Xc, yc
model=(;type=modaldecisiontree, params=(;conditions=[maximum]))
preprocess=(rng=Xoshiro(1), train_ratio=0.7)
# in _prepare_dataset
rng = hasproperty(preprocess, :rng) ? preprocess.rng : TaskLocalRNG()
mach = modelset(X, y, model; rng)

NUMERIC_TYPE = Union{Number, Missing}
@btime all(col -> typeof(col) <: Vector{<:VALID_NUMBER}, eachcol(Xc))
# 190.091 ns (2 allocations: 32 bytes)

@btime all(T -> T <: VALID_NUMBER, eltype.(eachcol(Xc)))
@btime all(T -> isa(T, VALID_NUMBER), eltype.(eachcol(Xc)))
# 1.506 μs (13 allocations: 496 bytes)

@btime all(T -> isa(T, Vector{<:VALID_NUMBER}), eachcol(Xc))

@btime all(col -> eltype(col) <: Number, eachcol(Xts))
# 170.954 ns (2 allocations: 32 bytes)

@btime all(T -> T <: Number, eltype.(eachcol(Xts)))

@btime begin
    all(eachcol(Xc)) do col
        all(x -> isa(x, VALID_NUMBER), col)
    end
end

@btime begin
    all(eachcol(Xts)) do col
        all(x -> isa(x, Number) || ismissing(x), col)
    end
end

@btime all(x -> isa(x, Number) || ismissing(x), Xc)

struct t1{T<:AbstractDataFrame}
    data  :: T
    treat :: Symbol
    win   :: NamedTuple
end

is_numeric_dataframe(df::AbstractDataFrame) = all(T -> T <: NUMERIC_TYPE, eltype.(eachcol(df)))
@inline is_all_numeric(df::AbstractDataFrame) = all(T -> <:(T, Number), eltype.(eachcol(df)))
@inline is_all_numeric_with_missing(df::AbstractDataFrame) = all(T -> T <: Union{Number, Missing}, eltype.(eachcol(df)))

@btime is_numeric_dataframe(Xc)
# 1.484 μs (11 allocations: 464 bytes)
@btime is_all_numeric(Xc)

@btime is_all_numeric_with_missing(Xc)
 win=(type=adaptivewindow, params=(nwindows=3, relative_overlap=0.1))