"""
    dataset.jl

Dataset construction and management utilities for SoleXplorer, providing high-level
interfaces for creating and manipulating datasets with modal logic support and
automatic model selection.

This module handles the creation of specialized dataset objects that encapsulate
machines, partitioning information, and treatment details for both propositional
and modal learning scenarios.
"""

# ---------------------------------------------------------------------------- #
#                               Abstract Types                                 #
# ---------------------------------------------------------------------------- #

"""
    AbstractDataSet

Abstract supertype for all dataset objects in SoleXplorer.

Concrete subtypes include:
- [`PropositionalDataSet`](@ref): For standard ML algorithms with aggregated features
- [`ModalDataSet`](@ref): For modal logic algorithms with temporal structure preservation
"""
abstract type AbstractDataSet end

# ---------------------------------------------------------------------------- #
#                                   Types                                      #
# ---------------------------------------------------------------------------- #

"""
    Modal = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}

Type alias for models that support modal logic and temporal reasoning.
These models can work directly with multidimensional time series data
without requiring feature aggregation.
"""
const Modal  = Union{ModalDecisionTree, ModalRandomForest, ModalAdaBoost}

"""
    Tuning = Union{Nothing, MLJTuning.TuningStrategy}

Type alias for hyperparameter tuning strategies, allowing either no tuning
(`Nothing`) or any MLJ tuning strategy.
"""
const Tuning = Union{Nothing, MLJTuning.TuningStrategy}

"""
    OptAggregationInfo = Optional{AggregationInfo}

Optional aggregation information for feature extraction from multidimensional data.
"""
const OptAggregationInfo = Optional{AggregationInfo}

"""
    OptVector = Optional{AbstractVector}

Optional vector type, typically used for sample weights or similar optional parameters.
"""
const OptVector = Optional{AbstractVector}

# ---------------------------------------------------------------------------- #
#                                  Defaults                                    #
# ---------------------------------------------------------------------------- #

"""
    _DefaultModel(y::AbstractVector)::MLJ.Model

Return a default model appropriate for the target variable type.

# Arguments
- `y::AbstractVector`: Target variable vector

# Returns
- `DecisionTreeClassifier()` if `eltype(y) <: CLabel` (classification)
- `DecisionTreeRegressor()` if `eltype(y) <: RLabel` (regression)

# Throws
- `ArgumentError`: If the target type is not supported

This function is used when no explicit model is provided to `setup_dataset`,
automatically selecting between classification and regression based on the
target variable's element type from the Sole ecosystem.
"""
function _DefaultModel(y::AbstractVector)::MLJ.Model
    if     eltype(y) <: CLabel
        return DecisionTreeClassifier()
    elseif eltype(y) <: RLabel
        return DecisionTreeRegressor()
    else
        throw(ArgumentError("Unsupported type for y: $(eltype(y))"))
    end
end

# ---------------------------------------------------------------------------- #
#                                 Utilities                                    #
# ---------------------------------------------------------------------------- #

"""
    set_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model

Set the random number generator for a model that supports it.

# Arguments
- `m::MLJ.Model`: The model to modify
- `rng::AbstractRNG`: The random number generator to assign

# Returns
- The modified model with `rng` field set

This function mutates the model's `rng` field if it exists, ensuring
reproducible results across training sessions.
"""
function set_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    m.rng = rng
    return m
end

"""
    set_rng!(r::MLJ.ResamplingStrategy, rng::AbstractRNG)::ResamplingStrategy

Set the random number generator for a resampling strategy.

# Arguments
- `r::MLJ.ResamplingStrategy`: The resampling strategy to modify
- `rng::AbstractRNG`: The random number generator to assign

# Returns
- A new resampling strategy instance with the specified RNG

Creates a new instance of the resampling strategy with the same parameters
but using the provided random number generator.
"""
function set_rng!(r::MLJ.ResamplingStrategy, rng::AbstractRNG)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (rng=rng,))...)
end

"""
    set_tuning_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model

Set random number generators for tuning-related components of a model.

# Arguments
- `m::MLJ.Model`: The model (typically a `TunedModel`) to modify
- `rng::AbstractRNG`: The random number generator to assign

# Returns
- The modified model with RNGs set for tuning and resampling components

This function ensures that both the tuning strategy and resampling strategy
use the same RNG for reproducible hyperparameter optimization.
"""
function set_tuning_rng!(m::MLJ.Model, rng::AbstractRNG)::MLJ.Model
    hasproperty(m.tuning, :rng) && (m.tuning.rng = rng)
    hasproperty(m.resampling, :rng) && (m.resampling = set_rng!(m.resampling, rng))
    return m
end

"""
    set_fraction_train!(r::ResamplingStrategy, train_ratio::Real)::ResamplingStrategy

Set the training fraction for a resampling strategy.

# Arguments
- `r::ResamplingStrategy`: The resampling strategy to modify
- `train_ratio::Real`: The fraction of data to use for training (0.0 to 1.0)

# Returns
- A new resampling strategy with the specified training fraction

Primarily used with `Holdout` resampling to specify the train/test split ratio.
"""
function set_fraction_train!(r::ResamplingStrategy, train_ratio::Real)::ResamplingStrategy
    typeof(r)(merge(MLJ.params(r), (fraction_train=train_ratio,))...)
end

"""
    set_conditions!(m::MLJ.Model, conditions::Tuple{Vararg{Base.Callable}})::MLJ.Model

Set logical conditions (features) for modal decision tree models.

# Arguments
- `m::MLJ.Model`: The modal model to modify
- `conditions::Tuple{Vararg{Base.Callable}}`: Feature functions for modal logic

# Returns
- The modified model with conditions set

This function is specifically for ModalDecisionTrees package models that
require features to be passed as model parameters rather than applied
to the data beforehand.
"""
function set_conditions!(m::MLJ.Model, conditions::Tuple{Vararg{Base.Callable}})::MLJ.Model
    m.conditions = Function[conditions...]
    return m
end

"""
    code_dataset!(X::AbstractDataFrame)

In-place encoding of non-numeric columns in a DataFrame to numeric codes.

# Arguments
- `X::AbstractDataFrame`: The DataFrame to encode

# Returns
- The modified DataFrame with categorical columns converted to numeric codes

Converts non-numeric columns to categorical arrays and then to level codes,
making the data suitable for standard ML algorithms that require numeric input.
"""
function code_dataset!(X::AbstractDataFrame)
    for (name, col) in pairs(eachcol(X))
        if !(eltype(col) <: Number)
            X_coded = MLJ.levelcode.(categorical(col)) 
            X[!, name] = X_coded
        end
    end
    
    return X
end

"""
    code_dataset!(y::AbstractVector)

In-place encoding of non-numeric target vector to numeric codes.

# Arguments
- `y::AbstractVector`: The target vector to encode

# Returns
- The modified vector with categorical values converted to numeric codes

Handles symbol-to-string conversion before categorical encoding to ensure
compatibility with MLJ's categorical handling system.
"""
function code_dataset!(y::AbstractVector)
    if !(eltype(y) <: Number)
        eltype(y) <: Symbol && (y = string.(y))
        y = MLJ.levelcode.(categorical(y)) 
    end
    
    return y
end

"""
    code_dataset!(X::AbstractDataFrame, y::AbstractVector)

Convenience method to encode both features and target simultaneously.

# Arguments
- `X::AbstractDataFrame`: The feature DataFrame to encode
- `y::AbstractVector`: The target vector to encode

# Returns
- Tuple of (encoded_X, encoded_y)
"""
code_dataset!(X::AbstractDataFrame, y::AbstractVector) = code_dataset!(X), code_dataset!(y)

"""
    range(field::Union{Symbol,Expr}; kwargs...)

Wrapper for MLJ.range in hyperparameter tuning contexts.

# Arguments
- `field::Union{Symbol,Expr}`: Model field to tune
- `kwargs...`: Range specification arguments

# Returns
- Tuple of (field, kwargs) for later processing by tuning setup

This function provides a more convenient syntax for specifying hyperparameter
ranges that will be converted to proper MLJ ranges once the model is available.
"""
Base.range(field::Union{Symbol,Expr}; kwargs...) = field, kwargs...

"""
    treat2aggr(t::TreatmentInfo)::AggregationInfo

Convert treatment information to aggregation information.

# Arguments
- `t::TreatmentInfo`: Treatment information containing features and window parameters

# Returns
- `AggregationInfo`: Aggregation information for feature extraction

Used internally to convert between different representations of how
multidimensional data should be processed.
"""
treat2aggr(t::TreatmentInfo)::AggregationInfo = 
    AggregationInfo(t.features, t.winparams)

# ---------------------------------------------------------------------------- #
#                          Dataset Type Definitions                            #
# ---------------------------------------------------------------------------- #

"""
    PropositionalDataSet{M} <: AbstractDataSet

Dataset for standard (propositional) machine learning algorithms.

# Fields
- `mach::MLJ.Machine`: The underlying MLJ machine
- `pidxs::Vector{PartitionIdxs}`: Partition indices for train/validation/test splits
- `pinfo::PartitionInfo`: Information about the partitioning strategy
- `ainfo::OptAggregationInfo`: Optional aggregation information for feature extraction

This dataset type is used when working with standard ML algorithms that require
tabular (propositional) data. Multidimensional time series are aggregated into
feature vectors using the specified aggregation strategy.
"""
mutable struct PropositionalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    ainfo   :: OptAggregationInfo
end

"""
    ModalDataSet{M} <: AbstractDataSet

Dataset for modal logic algorithms that work with temporal structures.

# Fields
- `mach::MLJ.Machine`: The underlying MLJ machine
- `pidxs::Vector{PartitionIdxs}`: Partition indices for train/validation/test splits
- `pinfo::PartitionInfo`: Information about the partitioning strategy
- `tinfo::TreatmentInfo`: Information about temporal data treatment

This dataset type preserves the temporal structure of multidimensional data,
making it suitable for modal decision trees and other algorithms that can
reason about temporal relationships directly.
"""
mutable struct ModalDataSet{M} <: AbstractDataSet
    mach    :: MLJ.Machine
    pidxs   :: Vector{PartitionIdxs}
    pinfo   :: PartitionInfo
    tinfo   :: TreatmentInfo
end

"""
    DataSet(mach, pidxs, pinfo; tinfo=nothing)

Construct an appropriate dataset type based on treatment information.

# Arguments
- `mach::MLJ.Machine{M}`: The MLJ machine
- `pidxs::Vector{PartitionIdxs}`: Partition indices
- `pinfo::PartitionInfo`: Partition information
- `tinfo::Union{TreatmentInfo, Nothing}`: Optional treatment information

# Returns
- `PropositionalDataSet{M}` if no treatment info or aggregation treatment
- `ModalDataSet{M}` if treatment is `:reducesize`

This constructor automatically determines the appropriate dataset type based on
whether temporal structure should be preserved (modal) or aggregated (propositional).
"""
function DataSet(
    mach    :: MLJ.Machine{M},
    pidxs   :: Vector{PartitionIdxs},
    pinfo   :: PartitionInfo;
    tinfo   :: Union{TreatmentInfo, Nothing}=nothing
) where {M<:MLJ.Model}
    isnothing(tinfo) ?
        PropositionalDataSet{M}(mach, pidxs, pinfo, nothing) : begin
            if tinfo.treatment == :reducesize
                ModalDataSet{M}(mach, pidxs, pinfo, tinfo)
            else
                ainfo = treat2aggr(tinfo)
                PropositionalDataSet{M}(mach, pidxs, pinfo, ainfo)
            end
        end
end

"""
    EitherDataSet = Union{PropositionalDataSet, ModalDataSet}

Type alias for either dataset type, useful for functions that work with both.
"""
const EitherDataSet = Union{PropositionalDataSet, ModalDataSet}

# ---------------------------------------------------------------------------- #
#                                Constructors                                  #
# ---------------------------------------------------------------------------- #

"""
    _prepare_dataset(X, y, w=nothing; kwargs...)::AbstractDataSet

Internal function to prepare and construct a dataset with all preprocessing applied.

# Arguments
- `X::AbstractDataFrame`: Feature data
- `y::AbstractVector`: Target variable
- `w::OptVector=nothing`: Optional sample weights

# Keyword Arguments
- `model::MLJ.Model=_DefaultModel(y)`: Model to use for training
- `resample::ResamplingStrategy=Holdout(shuffle=true)`: Resampling strategy
- `train_ratio::Real=0.7`: Fraction of data for training
- `valid_ratio::Real=0.0`: Fraction of data for validation
- `rng::AbstractRNG=TaskLocalRNG()`: Random number generator
- `win::WinFunction=AdaptiveWindow(nwindows=3, relative_overlap=0.1)`: Windowing function
- `features::Tuple{Vararg{Base.Callable}}=(maximum, minimum)`: Feature extraction functions
- `modalreduce::Base.Callable=mean`: Reduction function for modal algorithms
- `tuning::NamedTuple=NamedTuple()`: Hyperparameter tuning specification

# Returns
- `AbstractDataSet`: Either `PropositionalDataSet` or `ModalDataSet`

This function handles the complete pipeline of dataset preparation including:
1. Model configuration and RNG propagation
2. Multidimensional data treatment (aggregation vs. modal reduction)
3. Data partitioning and resampling setup
4. Hyperparameter tuning configuration
5. Machine construction with appropriate caching
"""
function _prepare_dataset(
    X             :: AbstractDataFrame,
    y             :: AbstractVector,
    w             :: OptVector               = nothing;
    model         :: MLJ.Model               = _DefaultModel(y),
    resample      :: ResamplingStrategy      = Holdout(shuffle=true),
    train_ratio   :: Real                    = 0.7,
    valid_ratio   :: Real                    = 0.0,
    rng           :: AbstractRNG             = TaskLocalRNG(),
    win           :: WinFunction             = AdaptiveWindow(nwindows=3, relative_overlap=0.1),
    features      :: Tuple{Vararg{Base.Callable}} = (maximum, minimum),
    modalreduce   :: Base.Callable           = mean,
    tuning        :: NamedTuple              = NamedTuple()
)::AbstractDataSet
    # propagate user rng to every field that needs it
    # model
    hasproperty(model,    :rng) && (model    = set_rng!(model,    rng))
    hasproperty(resample, :rng) && (resample = set_rng!(resample, rng))

    # ModalDecisionTrees package needs features to be passed in model params
    hasproperty(model, :features) && (model = set_conditions!(model, features))
    # Holdout resampling needs to setup fraction_train parameters
    resample isa Holdout && (resample = set_fraction_train!(resample, train_ratio))

    # Handle multidimensional datasets:
    # Decision point: use standard ML algorithms (requiring feature aggregation)
    # or modal logic algorithms (preserving temporal structure)
    # Standard algorithms: reduce to numeric datasets via feature extraction over windows
    # Modal algorithms: reduce data size for computational efficiency via modalreduce parameter
    if X[1, 1] isa AbstractArray
        treat = model isa Modal ? :reducesize : :aggregate
        X, tinfo = treatment(X; win, features, treat, modalreduce)
    else
        X = code_dataset!(X)
        tinfo = nothing
    end

    ttpairs, pinfo = partition(y; resample, train_ratio, valid_ratio, rng)

    isempty(tuning) || begin
        if !(tuning.range isa MLJ.NominalRange)
            # Convert SX.range to MLJ.range now that model is available
            range = tuning.range isa Tuple{Vararg{Tuple}} ? tuning.range : (tuning.range,)
            range = collect(MLJ.range(model, r[1]; r[2:end]...) for r in range)
            tuning = merge(tuning, (range=range,))
        end

        model = MLJ.TunedModel(model; tuning...)

        # Set the model to use the same rng as the dataset
        model = set_tuning_rng!(model, rng)
    end

    mach = isnothing(w) ? MLJ.machine(model, X, y) : MLJ.machine(model, X, y, w)
    
    DataSet(mach, ttpairs, pinfo; tinfo)
end

"""
    setup_dataset(args...; kwargs...)

High-level interface for dataset creation with automatic preprocessing.

This is the main entry point for creating datasets in SoleXplorer. It provides
a convenient interface that handles all the complexity of preparing data for
both standard and modal machine learning algorithms.

See [`_prepare_dataset`](@ref) for detailed parameter descriptions.

# Example
```julia
# Standard classification dataset
X, y = load_iris()
dataset = setup_dataset(X, y, model=DecisionTreeClassifier())

# Modal time series dataset  
X, y = load_temporal_data()
dataset = setup_dataset(X, y, 
                       model=ModalDecisionTree(),
                       win=MovingWindow(window_size=10),
                       features=(maximum, minimum, mean))
```
"""
setup_dataset(args...; kwargs...) = _prepare_dataset(args...; kwargs...)

"""
    setup_dataset(X::AbstractDataFrame, y::Symbol; kwargs...)::AbstractDataSet

Convenience method when target variable is a column in the feature DataFrame.

# Arguments
- `X::AbstractDataFrame`: DataFrame containing both features and target
- `y::Symbol`: Column name of the target variable
- `kwargs...`: Additional arguments passed to `setup_dataset`

# Returns
- `AbstractDataSet`: Configured dataset with target column separated

# Example
```julia
df = DataFrame(feature1=rand(100), feature2=rand(100), target=rand(["A", "B"], 100))
dataset = setup_dataset(df, :target)
```
"""
function setup_dataset(
    X::AbstractDataFrame,
    y::Symbol;
    kwargs...
)::AbstractDataSet
    setup_dataset(X[!, Not(y)], X[!, y]; kwargs...)
end

"""
    length(ds::EitherDataSet)

Return the number of data partitions in the dataset.

# Arguments
- `ds::EitherDataSet`: The dataset

# Returns
- `Int`: Number of partition indices

This corresponds to the number of cross-validation folds or train/test splits
configured for the dataset.
"""
Base.length(ds::EitherDataSet) = length(ds.pidxs)

"""
    get_y_test(ds::EitherDataSet)::AbstractVector

Extract test target values for each partition in the dataset.

# Arguments
- `ds::EitherDataSet`: The dataset

# Returns
- `Vector`: Vector of target value vectors, one for each partition's test set

This function is useful for extracting the ground truth values for evaluation
across multiple data partitions.
"""
get_y_test(ds::EitherDataSet)::AbstractVector = 
    [@views ds.mach.args[2].data[ds.pidxs[i].test] for i in 1:length(ds)]

"""
    get_mach_model(ds::EitherDataSet)::MLJ.Model

Extract the model from the dataset's machine.

# Arguments
- `ds::EitherDataSet`: The dataset

# Returns
- `MLJ.Model`: The model bound to the dataset's machine

Provides access to the underlying model for inspection or modification.
"""
get_mach_model(ds::EitherDataSet)::MLJ.Model = ds.mach.model