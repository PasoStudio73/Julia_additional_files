abstract type ModalEvoType end

abstract type ModalGradientRegression <: ModalEvoType            end
abstract type ModalMLE2P              <: ModalEvoType            end # 2-parameters max-likelihood

abstract type ModalMSE                <: ModalGradientRegression end
abstract type ModalLogLoss            <: ModalGradientRegression end
abstract type ModalPoisson            <: ModalGradientRegression end
abstract type ModalGamma              <: ModalGradientRegression end
abstract type ModalTweedie            <: ModalGradientRegression end
abstract type ModalMLogLoss           <: ModalEvoType            end
abstract type ModalGaussianMLE        <: ModalMLE2P              end
abstract type ModalLogisticMLE        <: ModalMLE2P              end
abstract type ModalQuantile           <: ModalEvoType            end
abstract type ModalL1                 <: ModalEvoType            end

# Converts ModalMSE -> :mse
# const _type2loss_dict = Dict(
#     ModalMSE => :mse,
#     ModalLogLoss => :logloss,
#     ModalPoisson => :poisson,
#     ModalGamma => :gamma,
#     ModalTweedie => :tweedie,
#     ModalMLogLoss => :mlogloss,
#     ModalGaussianMLE => :gaussian_mle,
#     ModalLogisticMLE => :logistic_mle,
#     ModalQuantile => :quantile,
#     ModalL1 => :l1,
# )
# _type2loss(L::Type) = _type2loss_dict[L]

# # make a Random Number Generator object
# mk_rng(rng::AbstractRNG) = rng
# mk_rng(int::Integer) = Random.MersenneTwister(int)

# mutable struct EvoTreeRegressor{L<:ModelType} <: MMI.Deterministic
#     nrounds::Int
#     L2::Float64
#     lambda::Float64
#     gamma::Float64
#     eta::Float64
#     max_depth::Int
#     min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
#     rowsample::Float64 # subsample
#     colsample::Float64
#     nbins::Int
#     alpha::Float64
#     monotone_constraints::Any
#     tree_type::String
#     rng::Any
# end

# function EvoTreeRegressor(; kwargs...)

#     # defaults arguments
#     args = Dict{Symbol,Any}(
#         :loss => :mse,
#         :nrounds => 100,
#         :L2 => 0.0,
#         :lambda => 0.0,
#         :gamma => 0.0, # min gain to split
#         :eta => 0.1, # learning rate
#         :max_depth => 6,
#         :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
#         :rowsample => 1.0,
#         :colsample => 1.0,
#         :nbins => 64,
#         :alpha => 0.5,
#         :monotone_constraints => Dict{Int,Int}(),
#         :tree_type => "binary",
#         :rng => 123,
#     )

#     args_override = intersect(keys(args), keys(kwargs))
#     for arg in args_override
#         args[arg] = kwargs[arg]
#     end

#     args[:rng] = mk_rng(args[:rng])
#     args[:loss] = Symbol(args[:loss])

#     if args[:loss] == :mse
#         L = ModalMSE
#     elseif args[:loss] == :linear
#         L = ModalMSE
#     elseif args[:loss] == :logloss
#         L = ModalLogLoss
#     elseif args[:loss] == :logistic
#         L = ModalLogLoss
#     elseif args[:loss] == :gamma
#         L = ModalGamma
#     elseif args[:loss] == :tweedie
#         L = ModalTweedie
#     elseif args[:loss] == :l1
#         L = ModalL1
#     elseif args[:loss] == :quantile
#         L = ModalQuantile
#     else
#         error(
#             "Invalid loss: $(args[:loss]). Only [`:mse`, `:logloss`, `:gamma`, `:tweedie`, `:l1`, `:quantile`] are supported by EvoTreeRegressor.",
#         )
#     end

#     check_args(args)

#     model = EvoTreeRegressor{L}(
#         args[:nrounds],
#         args[:L2],
#         args[:lambda],
#         args[:gamma],
#         args[:eta],
#         args[:max_depth],
#         args[:min_weight],
#         args[:rowsample],
#         args[:colsample],
#         args[:nbins],
#         args[:alpha],
#         args[:monotone_constraints],
#         args[:tree_type],
#         args[:rng],
#     )

#     return model
# end

# function EvoTreeRegressor{L}(; kwargs...) where {L}
#     EvoTreeRegressor(; loss=_type2loss(L), kwargs...)
# end

# mutable struct EvoTreeCount{L<:ModelType} <: MMI.Probabilistic
#     nrounds::Int
#     L2::Float64
#     lambda::Float64
#     gamma::Float64
#     eta::Float64
#     max_depth::Int
#     min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
#     rowsample::Float64 # subsample
#     colsample::Float64
#     nbins::Int
#     alpha::Float64
#     monotone_constraints::Any
#     tree_type::String
#     rng::Any
# end

# function EvoTreeCount(; kwargs...)

#     # defaults arguments
#     args = Dict{Symbol,Any}(
#         :nrounds => 100,
#         :L2 => 0.0,
#         :lambda => 0.0,
#         :gamma => 0.0, # min gain to split
#         :eta => 0.1, # learning rate
#         :max_depth => 6,
#         :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
#         :rowsample => 1.0,
#         :colsample => 1.0,
#         :nbins => 64,
#         :alpha => 0.5,
#         :monotone_constraints => Dict{Int,Int}(),
#         :tree_type => "binary",
#         :rng => 123,
#     )

#     args_override = intersect(keys(args), keys(kwargs))
#     for arg in args_override
#         args[arg] = kwargs[arg]
#     end

#     args[:rng] = mk_rng(args[:rng])
#     L = ModalPoisson
#     check_args(args)

#     model = EvoTreeCount{L}(
#         args[:nrounds],
#         args[:L2],
#         args[:lambda],
#         args[:gamma],
#         args[:eta],
#         args[:max_depth],
#         args[:min_weight],
#         args[:rowsample],
#         args[:colsample],
#         args[:nbins],
#         args[:alpha],
#         args[:monotone_constraints],
#         args[:tree_type],
#         args[:rng],
#     )

#     return model
# end

# function EvoTreeCount{L}(; kwargs...) where {L}
#     EvoTreeCount(; kwargs...)
# end

mutable struct ModalEvoTreeClassifier{L<:ModalEvoType} <: MMI.Probabilistic
    nrounds     ::Int
    L2          ::Float64
    lambda      ::Float64
    gamma       ::Float64
    eta         ::Float64
    max_depth   ::Int
    min_weight  ::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample   ::Float64 # subsample
    colsample   ::Float64
    nbins       ::Int
    alpha       ::Float64
    tree_type   ::String
    rng         ::Any
end

function ModalEvoTreeClassifier(; kwargs...)
    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds    => 100,
        :L2         => 0.0,
        :lambda     => 0.0,
        :gamma      => 0.0, # min gain to split
        :eta        => 0.1, # learning rate
        :max_depth  => 6,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample  => 1.0,
        :colsample  => 1.0,
        :nbins      => 64,
        :alpha      => 0.5,
        :tree_type  => "binary",
        :rng        => Random.TaskLocalRNG(),
    )

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    # args[:rng] = mk_rng(args[:rng])
    L = ModalMLogLoss
    Evo.check_args(args)

    model = ModalEvoTreeClassifier{L}(
        args[:nrounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:nbins],
        args[:alpha],
        args[:tree_type],
        args[:rng],
    )

    return model
end

function ModalEvoTreeClassifier{L}(; kwargs...) where {L}
    ModalEvoTreeClassifier(; kwargs...)
end

# mutable struct EvoTreeMLE{L<:ModelType} <: MMI.Probabilistic
#     nrounds::Int
#     L2::Float64
#     lambda::Float64
#     gamma::Float64
#     eta::Float64
#     max_depth::Int
#     min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
#     rowsample::Float64 # subsample
#     colsample::Float64
#     nbins::Int
#     alpha::Float64
#     monotone_constraints::Any
#     tree_type::String
#     rng::Any
# end

# function EvoTreeMLE(; kwargs...)

#     # defaults arguments
#     args = Dict{Symbol,Any}(
#         :loss => :gaussian_mle,
#         :nrounds => 100,
#         :L2 => 0.0,
#         :lambda => 0.0,
#         :gamma => 0.0, # min gain to split
#         :eta => 0.1, # learning rate
#         :max_depth => 6,
#         :min_weight => 8.0, # minimal weight, different from xgboost (but same for linear)
#         :rowsample => 1.0,
#         :colsample => 1.0,
#         :nbins => 64,
#         :alpha => 0.5,
#         :monotone_constraints => Dict{Int,Int}(),
#         :tree_type => "binary",
#         :rng => 123,
#     )

#     args_override = intersect(keys(args), keys(kwargs))
#     for arg in args_override
#         args[arg] = kwargs[arg]
#     end

#     args[:rng] = mk_rng(args[:rng])
#     args[:loss] = Symbol(args[:loss])

#     if args[:loss] in [:gaussian, :gaussian_mle]
#         L = ModalGaussianMLE
#     elseif args[:loss] in [:logistic, :logistic_mle]
#         L = ModalLogisticMLE
#     else
#         error(
#             "Invalid loss: $(args[:loss]). Only `:gaussian_mle` and `:logistic_mle` are supported by EvoTreeMLE.",
#         )
#     end

#     check_args(args)

#     model = EvoTreeMLE{L}(
#         args[:nrounds],
#         args[:L2],
#         args[:lambda],
#         args[:gamma],
#         args[:eta],
#         args[:max_depth],
#         args[:min_weight],
#         args[:rowsample],
#         args[:colsample],
#         args[:nbins],
#         args[:alpha],
#         args[:monotone_constraints],
#         args[:tree_type],
#         args[:rng],
#     )

#     return model
# end

# function EvoTreeMLE{L}(; kwargs...) where {L}
#     if L == ModalGaussianMLE
#         loss = :gaussian_mle
#     elseif L == ModalLogisticMLE
#         loss = :logistic_mle
#     end
#     EvoTreeMLE(; loss=loss, kwargs...)
# end


# mutable struct EvoTreeGaussian{L<:ModelType} <: MMI.Probabilistic
#     nrounds::Int
#     L2::Float64
#     lambda::Float64
#     gamma::Float64
#     eta::Float64
#     max_depth::Int
#     min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
#     rowsample::Float64 # subsample
#     colsample::Float64
#     nbins::Int
#     alpha::Float64
#     monotone_constraints::Any
#     tree_type::String
#     rng::Any
# end
# function EvoTreeGaussian(; kwargs...)

#     # defaults arguments
#     args = Dict{Symbol,Any}(
#         :nrounds => 100,
#         :L2 => 0.0,
#         :lambda => 0.0,
#         :gamma => 0.0, # min gain to split
#         :eta => 0.1, # learning rate
#         :max_depth => 6,
#         :min_weight => 8.0, # minimal weight, different from xgboost (but same for linear)
#         :rowsample => 1.0,
#         :colsample => 1.0,
#         :nbins => 64,
#         :alpha => 0.5,
#         :monotone_constraints => Dict{Int,Int}(),
#         :tree_type => "binary",
#         :rng => 123,
#     )

#     args_override = intersect(keys(args), keys(kwargs))
#     for arg in args_override
#         args[arg] = kwargs[arg]
#     end

#     args[:rng] = mk_rng(args[:rng])
#     L = ModalGaussianMLE
#     check_args(args)

#     model = EvoTreeGaussian{L}(
#         args[:nrounds],
#         args[:L2],
#         args[:lambda],
#         args[:gamma],
#         args[:eta],
#         args[:max_depth],
#         args[:min_weight],
#         args[:rowsample],
#         args[:colsample],
#         args[:nbins],
#         args[:alpha],
#         args[:monotone_constraints],
#         args[:tree_type],
#         args[:rng],
#     )

#     return model
# end

# function EvoTreeGaussian{L}(; kwargs...) where {L}
#     EvoTreeGaussian(; kwargs...)
# end

const ModalEvoTypes{L} = Union{
    # ModalEvoTreeRegressor{L},
    # ModalEvoTreeCount{L},
    ModalEvoTreeClassifier{L},
    # ModalEvoTreeGaussian{L},
    # ModalEvoTreeMLE{L},
}

_get_struct_loss(::ModalEvoTypes{L}) where {L} = L

function Base.show(io::IO, config::ModalEvoTypes)
    println(io, "$(typeof(config))")
    for fname in fieldnames(typeof(config))
        println(io, " - $fname: $(getfield(config, fname))")
    end
end

# """
#     check_parameter(::Type{<:T}, value, min_value::Real, max_value::Real, label::Symbol) where {T<:Number}

# Check model parameter if it's valid
# """
# function check_parameter(::Type{<:T}, value, min_value::Real, max_value::Real, label::Symbol) where {T<:Number}
#     min_value = max(typemin(T), min_value)
#     max_value = min(typemax(T), max_value)
#     try
#         convert(T, value)
#         @assert min_value <= value <= max_value
#     catch
#         error("Invalid value for parameter `$(string(label))`: $value. `$(string(label))` must be of type $T with value between $min_value and $max_value.")
#     end
# end

# """
#     check_args(args::Dict{Symbol,Any})

# Check model arguments if they are valid
# """
# function check_args(args::Dict{Symbol,Any})

#     # Check integer parameters
#     check_parameter(Int, args[:nrounds], 0, typemax(Int), :nrounds)
#     check_parameter(Int, args[:max_depth], 1, typemax(Int), :max_depth)
#     check_parameter(Int, args[:nbins], 2, 255, :nbins)

#     # check positive float parameters
#     check_parameter(Float64, args[:lambda], zero(Float64), typemax(Float64), :lambda)
#     check_parameter(Float64, args[:gamma], zero(Float64), typemax(Float64), :gamma)
#     check_parameter(Float64, args[:min_weight], zero(Float64), typemax(Float64), :min_weight)

#     # check bounded parameters
#     check_parameter(Float64, args[:alpha], zero(Float64), one(Float64), :alpha)
#     check_parameter(Float64, args[:rowsample], eps(Float64), one(Float64), :rowsample)
#     check_parameter(Float64, args[:colsample], eps(Float64), one(Float64), :colsample)
#     check_parameter(Float64, args[:eta], zero(Float64), typemax(Float64), :eta)

#     try
#         tree_type = string(args[:tree_type])
#         @assert tree_type ∈ ["binary", "oblivious"]
#     catch
#         error("Invalid input for `tree_type` parameter: `$(args[:tree_type])`. Must be of one of `binary` or `oblivious`")
#     end

# end

# """
#     check_args(model::EvoTypes{L}) where {L}

# Check model arguments if they are valid (eg, after mutation when tuning hyperparams)
# Note: does not check consistency of model type and loss selected
# """
# function check_args(model::EvoTypes{L}) where {L}

#     # Check integer parameters
#     check_parameter(Int, model.max_depth, 1, typemax(Int), :max_depth)
#     check_parameter(Int, model.nrounds, 0, typemax(Int), :nrounds)
#     check_parameter(Int, model.nbins, 2, 255, :nbins)

#     # check positive float parameters
#     check_parameter(Float64, model.lambda, zero(Float64), typemax(Float64), :lambda)
#     check_parameter(Float64, model.gamma, zero(Float64), typemax(Float64), :gamma)
#     check_parameter(Float64, model.min_weight, zero(Float64), typemax(Float64), :min_weight)

#     # check bounded parameters
#     check_parameter(Float64, model.alpha, zero(Float64), one(Float64), :alpha)
#     check_parameter(Float64, model.rowsample, eps(Float64), one(Float64), :rowsample)
#     check_parameter(Float64, model.colsample, eps(Float64), one(Float64), :colsample)
#     check_parameter(Float64, model.eta, zero(Float64), typemax(Float64), :eta)

#     try
#         tree_type = string(model.tree_type)
#         @assert tree_type ∈ ["binary", "oblivious"]
#     catch
#         error("Invalid input for `tree_type` parameter: `$(model.tree_type)`. Must be of one of `binary` or `oblivious`")
#     end
# end