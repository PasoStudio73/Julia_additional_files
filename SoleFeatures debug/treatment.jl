# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
# check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real,AbstractArray{<:Real}}, eachcol(df))
# hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

using SoleBase, SoleData
using StatsBase, Catch22
using CategoricalArrays, DataFrames
using Random

include("/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/src/utils/features_set.jl")
include("/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/src/dataset/dataset_structs.jl")

X, y = SoleData.load_arff_dataset("NATOPS")

# ---------------------------------------------------------------------------- #
#                              check dimensions                                #
# ---------------------------------------------------------------------------- #
"""
    _check_dimensions(X::DataFrame) -> Int

Internal function.
Check dimensionality of elements in DataFrame columns.
Currently supports only scalar values and time series (1-dimensional arrays).

# Returns
- `Int`: 0 for scalar elements, 1 for 1D array elements

# Throws
- `DimensionMismatch`: If elements have inconsistent dimensions
- `ArgumentError`: If elements have more than 1D
"""
function _check_dimensions(X::DataFrame)
    isempty(X) && return 0
    
    # Get reference dimensions from first element
    first_col = first(eachcol(X))
    ref_dims = ndims(first(first_col))
    
    # Early dimension check
    ref_dims > 1 && throw(ArgumentError("Elements more than 1D are not supported."))
    
    # Check all columns maintain same dimensionality
    all(col -> all(x -> ndims(x) == ref_dims, col), eachcol(X)) ||
        throw(DimensionMismatch("Inconsistent dimensions across elements"))
    
    return ref_dims
end

# ---------------------------------------------------------------------------- #
#                                 treatment                                    #
# ---------------------------------------------------------------------------- #
"""
    _treatment(X::DataFrame, vnames::AbstractVector{String}, treatment::Symbol, 
               features::AbstractVector{<:Base.Callable}, winparams::NamedTuple)

Internal function.
Processes the input DataFrame `X` based on the specified `treatment` type, 
either aggregating or reducing the size of the data. The function applies 
the given `features` to the columns specified by `vnames`, using window 
parameters defined in `winparams`.

# Arguments
- `X::DataFrame`: The input data to be processed.
- `vnames::AbstractVector{String}`: Names of the columns in `X` to be treated.
- `treatment::Symbol`: The type of treatment to apply, either `:aggregate` 
  or `:reducesize`.
- `features::AbstractVector{<:Base.Callable}`: Functions to apply to the 
  specified columns.
- `winparams::NamedTuple`: Parameters defining the windowing strategy, 
  including the type of window function.

# Returns
- `DataFrame`: A new DataFrame with the processed data.

# Throws
- `ArgumentError`: If `winparams` does not contain a valid `type`.
"""

function _treatment(
    X::DataFrame,
    vnames::AbstractVector{String},
    treatment::Symbol,
    features::AbstractVector{<:Base.Callable},
    winparams::NamedTuple
)
    # check parameters
    haskey(winparams, :type) || throw(ArgumentError("winparams must contain a type, $(keys(WIN_PARAMS))"))
    haskey(WIN_PARAMS, winparams.type) || throw(ArgumentError("winparams.type must be one of: $(keys(WIN_PARAMS))"))

    max_interval = maximum(length.(eachrow(X)))
    _wparams = NamedTuple(k => v for (k,v) in pairs(winparams) if k != :type)
    n_intervals = winparams.type(max_interval; _wparams...)

    # Initialize DataFrame
    if treatment == :aggregate        # propositional
        if n_intervals == 1
            valid_X = DataFrame([v => Float64[]
                                 for v in [string(f, "(", v, ")")
                                       for f in features for v in vnames]]
            )
        else
            valid_X = DataFrame([v => Float64[]
                                 for v in [string(f, "(", v, ")w", i)
                                       for f in features for v in vnames
                                       for i in 1:length(n_intervals)]]
            )
        end

    elseif treatment == :reducesize   # modal
        # valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
        valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])

    elseif treatment == :feature_selection
        if n_intervals == 1
            # valid_X = DataFrame([v => Float64[]
            valid_X = DataFrame([v => Feature[]
                for v in [string(f, "(", v, ")")
                    for f in features for v in vnames]]
            )
        else
            # valid_X = DataFrame([v => Float64[]
            valid_X = DataFrame([v => Feature[]
                for v in [string(f, "(", v, ")w", i)
                    for f in features for v in vnames
                    for i in 1:length(n_intervals)]]
            )
        end
    end

    # Fill DataFrame
    for row in eachrow(X)
        row_intervals = winparams.type(maximum(length.(collect(row))); _wparams...)
        # interval_dif is used in case we encounter a row with less intervals than the maximum
        interval_diff = length(n_intervals) - length(row_intervals)

        if treatment == :aggregate
            push!(valid_X, vcat([
                vcat([f(col[r]) for r in row_intervals],
                    # if interval_diff is positive, fill the rest with NaN
                    fill(NaN, interval_diff)) for col in row, f in features
                ]...)
            )
        elseif treatment == :reducesize
            f = haskey(_wparams, :reducefunc) ? _wparams.reducefunc : mean
            push!(valid_X, [
                vcat([f(col[r]) for r in row_intervals],
                    # if interval_diff is positive, fill the rest with NaN
                    fill(NaN, interval_diff)) for col in row
                ]
            )
        elseif treatment == :feature_selection
            push!(valid_X, vcat([
                vcat([
                    Feature(f(col[r]), vnames[i], Symbol(f), w) for (w, r) in enumerate(row_intervals)],
                    # if interval_diff is positive, fill the rest with NaN
                    fill(NaN, interval_diff)) for (i, col) in enumerate(row), f in features
                ]...)
            )
        end
    end

    return valid_X
end

function feature_selection_preprocess(
    X::DataFrame;
    vnames::Union{Vector{String}, Vector{Symbol}, Nothing}=nothing,
    features::Union{Vector{<:Base.Callable}, Nothing}=nothing,
    nwindows::Union{Int, Nothing}=nothing
)
    # check parameters
    isnothing(vnames) && (vnames = names(X))
    isnothing(features) && (features = DEFAULT_FE.features)
    treatment = :feature_selection
    _ = _check_dimensions(X)
    winparams = isnothing(nwindows) ? DEFAULT_FE_WINPARAMS : merge(DEFAULT_FE_WINPARAMS, (nwindows = nwindows,))

    _treatment(X, vnames, treatment, features, winparams)
end

a = feature_selection_preprocess(X)