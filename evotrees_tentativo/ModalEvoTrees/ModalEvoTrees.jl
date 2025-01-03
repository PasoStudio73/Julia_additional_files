module ModalEvoTrees

# export fit_evotree
# export EvoTreeRegressor,
#     EvoTreeCount,
#     EvoTreeClassifier,
#     EvoTreeMLE,
#     EvoTreeGaussian,
#     EvoTree,
#     Random

export ModalEvoTreeClassifier

using Base.Threads: @threads, @spawn, nthreads, threadid
using DataFrames
# using Statistics
using StatsBase: mean, sample, sample!, quantile, proportions
# using Random
# using Random: seed!, AbstractRNG
# using Distributions
# using Tables
using CategoricalArrays
# using Tables
# using BSON

# using NetworkLayout
# using RecipesBase

using EvoTrees
import EvoTrees as Evo
import EvoTrees: CPU, Logistic, Poisson, Gamma, Tweedie, MLogLoss, GaussianMLE, LogisticMLE

using MLJModelInterface
import MLJModelInterface as MMI
# import MLJModelInterface: fit, update, predict, schema
# import Base: convert

include("models.jl")

# include("structs.jl")
# include("loss.jl")
# include("eval.jl")
# include("predict.jl")
include("init.jl")
# include("subsample.jl")
include("fit-utils.jl")
# include("fit.jl")

# if !isdefined(Base, :get_extension)
#     include("../ext/EvoTreesCUDAExt/EvoTreesCUDAExt.jl")
# end

# include("callback.jl")
# include("importance.jl")
# include("plot.jl")
include("MLJ.jl")

# function save(model::EvoTree, path)
#     BSON.bson(path, Dict(:model => model))
# end

# function load(path)
#     m = BSON.load(path, @__MODULE__)
#     return m[:model]
# end

end # module
