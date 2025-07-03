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

using StaticArrays

@btime begin
    a = nrow(Xc)
    b = ncol(Xc)
    c = Array(Xc)
    D = SizedArray{Tuple{a,b}}(c)
end
# 4.008 μs (16 allocations: 5.32 KiB)

@btime begin
    D = SizedArray{Tuple{nrow(Xc),ncol(Xc)}}(Array(Xc))
end
# 4.000 μs (16 allocations: 5.32 KiB)

@btime begin
    D = Array(Xc)
end
# 1.724 μs (5 allocations: 4.85 KiB)

