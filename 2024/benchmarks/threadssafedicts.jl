using BenchmarkTools
# using ThreadSafeDicts

const RULES_PARAMS = Dict(
    :intrees => (prune_rules = true, pruning_s = nothing,),
    :refne => (L = 10, perc = 1.0,))

rule_params = Dict(
    :intrees => (prune_rules = true, pruning_s = nothing,),
    :refne => (L = 10, perc = 1.0,))

# The typed dictionary (RULES_TYPED) likely has a smaller memory footprint than the untyped one
# because Julia can optimize storage when it knows the exact types.
const RULES_TYPED = Dict{Symbol,NamedTuple}(
    :intrees => (prune_rules = true, pruning_s = nothing,),
    :refne => (L = 10, perc = 1.0,))

# const RULES_THR = ThreadSafeDict{Symbol,NamedTuple}([
#     :intrees => (prune_rules = true, pruning_s = nothing,),
#     :refne => (L = 10, perc = 1.0,)])

@btime RULES_PARAMS[:intrees]
# 3.222 ns (0 allocations: 0 bytes)
@btime rule_params[:intrees]
# 10.825 ns (0 allocations: 0 bytes)
@btime RULES_TYPED[:intrees]
# 3.196 ns (0 allocations: 0 bytes)
# @btime RULES_THR[:intrees]
# 18.249 ns (0 allocations: 0 bytes)


# Class Results
abstract type AbstractResults end

struct ClassResults <: AbstractResults
    accuracy   :: AbstractFloat
end

struct RegResults <: AbstractResults
    accuracy   :: AbstractFloat
end

const RESULTS = Dict{Symbol,DataType}(
    :classification => ClassResults,
    :regression     => RegResults
)

# MLJBase functions
using MLJBase

const RESAMPLE_PARAMS = Dict{DataType,NamedTuple}(
    CV => (nfolds = 6, shuffle = true,),
    Holdout => (fraction_train = 0.7, shuffle = true,))

# Tuning
using MLJ: Grid as grid, RandomSearch as randomsearch, LatinHypercube as latinhypercube
using MLJParticleSwarmOptimization: ParticleSwarm as particleswarm, AdaptiveParticleSwarm as adaptiveparticleswarm

const TUNING_PARAMS = Dict{DataType,NamedTuple}(
    particleswarm => (goal = nothing, resolution = 10,),
    randomsearch => (bounded = Distributions.Uniform, positive_unbounded = Distributions.Gamma,))

# dataset info
abstract type AbstractDatasetSetup end

struct DatasetInfo <: AbstractDatasetSetup
    algo        :: Symbol
    resample    :: Bool


function DatasetInfo(
        algo        :: Symbol,
        resample    :: Bool,
    )::DatasetInfo
        new(algo, resample)
    end
end
get_resample(test::DatasetInfo)::Bool = test.resample

struct D1 <: AbstractDatasetSetup
    data::DatasetInfo
end
get_resample(test::D1)::Bool = get_resample(test.data)
struct D2 <: AbstractDatasetSetup
    data::D1
end
get_resample(test::D2)::Bool = get_resample(test.data)
gr(test::D2)::Bool = test.data.data.resample

test = DatasetInfo(:intrees, true)
t1 = D1(test)
t2 = D2(t1)

@btime println(t2.data.data.resample)
# 2.874 μs (6 allocations: 160 bytes)
@btime println(get_resample(t2))
# 3.372 μs (3 allocations: 64 bytes)
@btime println(gr(t2))
# 2.826 μs (3 allocations: 64 bytes)


# ude get_
using SoleXplorer
using DataFrames
using MLJBase

X, y = @load_iris
X = DataFrame(X)

# original data struct
@btime modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(11)),
)
# 230.746 μs (2416 allocations: 169.49 KiB)

# optimized data struct
@btime modelset = symbolic_analysis(
    X, y;
    model=(;type=:decisiontree),
    preprocess=(;rng=Xoshiro(11)),
)