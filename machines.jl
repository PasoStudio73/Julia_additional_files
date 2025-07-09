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
a = symbolic_analysis(
    Xts, yts;
    model=(;type=modaldecisiontree, params=(;conditions=[maximum])),
    resample = (type=CV, params=(;shuffle=true)),
    measures=(log_loss, accuracy),
    preprocess=(rng=Xoshiro(1), train_ratio=0.7),
)

# Option 1: Compare key collections
@btime isempty(setdiff(keys(params), [:type, :params]))
# 219.844 ns (8 allocations: 528 bytes)

# Option 2: Check if a key exists
@btime keys(params) ⊆ (:type, :params)   # true
# 33.141 ns (0 allocations: 0 bytes)

@btime issubset(keys(params), (:type, :params)) 
# 33.377 ns (0 allocations: 0 bytes)

# Option 3: Use filter for more complex logic
@btime isempty(filter(k -> k ∉ (:type, :params), keys(params)))
# 197.700 ns (1 allocation: 32 bytes)

d = decisiontreeclassifier
m = modaldecisiontree

@btime hasproperty(d(), :features)
# 47.609 ns (1 allocation: 96 bytes)
@btime hasproperty(m(), :features)
# 1.776 μs (15 allocations: 688 bytes)
@btime hasfield(typeof(d()), :features)
# 39.718 ns (1 allocation: 96 bytes)
@btime hasfield(typeof(m()), :features)
# 1.757 μs (15 allocations: 688 bytes)

@btime :features ∈ propertynames(d)
@btime :features ∈ propertynames(m)

# 6. Using reflection with Base functions
@btime isdefined(d(), :features)
@btime isdefined(m(), :features)

# 9. Custom function combining multiple checks
function model_accepts_features(model_constructor)
    try
        # Try to create instance and check for features field
        instance = model_constructor()
        return hasfield(typeof(instance), :features)
    catch
        return false
    end
end

function validate_model(model::NamedTuple, rng::AbstractRNG)::MLJ.Model
    issubset(keys(model), (:type, :params)) || throw(ArgumentError("Unknown fields."))

    modeltype   = get(model, :type, nothing)
    isnothing(modeltype) && throw(ArgumentError("Each model specification must contain a 'type' field"))
    
    modelparams = get(model, :params, NamedTuple())

    if modeltype ∈ AVAIL_MODELS
        # Option 1: Use hasfield on the type (most efficient)
        model_instance = modeltype()
        @show hasfield(typeof(model_instance), :features)
        
        # Option 2: Check fieldnames (also efficient)
        model_fields = fieldnames(typeof(model_instance))
        @show :features ∈ model_fields
        
        # Option 3: Try-catch approach (safest)
        function accepts_rng(constructor)
            try
                return hasfield(typeof(constructor()), :rng)
            catch
                return false
            end
        end
        
        # Apply the check
        if accepts_rng(modeltype)
            modeltype(; rng, modelparams...)
        else
            modeltype(; modelparams...)
        end
    else
        throw(ArgumentError("Model $modeltype not found in available models"))
    end
end

# ---------------------------------------------------------------------------- #
#                           testing validate model                             #
# ---------------------------------------------------------------------------- #
@test_throws ArgumentError symbolic_analysis(
    Xc, yc;
    model=(;params=(min_samples_leaf=1,n_subfeatures=3)),
)

@test_throws ArgumentError symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier, invalid_param=1),
)

@test_throws ArgumentError symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier, invalid_param=1),
)

@test_throws MethodError symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier, params=(invalid=1,)),
)

# ---------------------------------------------------------------------------- #
#                         testing validate parameters                          #
# ---------------------------------------------------------------------------- #
test = symbolic_analysis(
    Xc, yc;
    model=(;type=modaldecisiontree),
)

test = symbolic_analysis(
    Xc, yc;
    model=(type=modaldecisiontree, params=(;conditions=[maximum])),
)

test = symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier),
)

test = symbolic_analysis(
    Xc, yc;
    model=(;type=decisiontreeclassifier),
    preprocess=(;rng=Xoshiro(1)),
)



# ---------------------------------------------------------------------------- #
#                              model constructor                               #
# ---------------------------------------------------------------------------- #
decisiontreeclassifier(; kwargs...) where T<:MLJ.Model = DecisionTreeClassifier{T}(; kwargs...)

# Option 1: Simple function (recommended for your use case)
decisiontreeclassifier{T}(; kwargs...) where T<:MLJ.Model = DecisionTreeClassifier(; kwargs...)

# Option 2: Function with type parameter (if you need it)
function decisiontreeclassifier(::Type{T}; kwargs...) where T<:MLJ.Model
    return DecisionTreeClassifier(; kwargs...)
end

# Option 3: Generic function with parametric dispatch
function modelconstructor(::T; kwargs...) where T<:MLJ.Model
    return T(; kwargs...)
end

modelconstructor(DecisionTreeClassifier; max_depth=5, min_samples_leaf=1)


# Option 4: Using a callable struct (more advanced)
struct ModelConstructor{T<:MLJModelInterface.Model}
    type::Type{T}
end
(m::ModelConstructor)(; kwargs...) = m.type(; kwargs...)

const decisiontreeclassifier = ModelConstructor(DecisionTreeClassifier)

# Option 5: Factory function pattern
function create_model_constructor(ModelType::Type{<:MLJ.Model})
    return (;kwargs...) -> ModelType(; kwargs...)
end

const decisiontreeclassifier = create_model_constructor(DecisionTreeClassifier)



struct ModelConstructor{T}
    type::Type{T}
end
(m::ModelConstructor)(; kwargs...) = m.type(; kwargs...)

const decisiontreeclassifier = ModelConstructor(DecisionTreeClassifier)

a=decisiontreeclassifier()

fieldnames(typeof(a))
# (:max_depth, :min_samples_leaf, :min_samples_split, :min_purity_increase, :n_subfeatures, :post_prune, :merge_purity_threshold, :display_depth, :feature_importance, :rng)

# Method 2: propertynames - Get property names (includes computed properties)
propertynames(a)
# (:max_depth, :min_samples_leaf, :min_samples_split, :min_purity_increase, :n_subfeatures, :post_prune, :merge_purity_threshold, :display_depth, :feature_importance, :rng)

# Method 3: Get field names and values together
@btime begin
    for field in fieldnames(typeof(a))
        println("$field: $(getfield(a, field))")
    end
end
# 59.039 μs (122 allocations: 4.34 KiB)

# Method 5: Create a custom display function
function display_fields(obj)
    println("Fields of $(typeof(obj)):")
    for field in fieldnames(typeof(obj))
        value = getfield(obj, field)
        println("  $field: $value")
    end
end

@btime display_fields(a)
# 66.347 μs (136 allocations: 4.89 KiB)

# Method 6: Get specific field values
getfield(a, :max_depth)
# -1

# Method 7: Check if a field exists
hasfield(typeof(a), :max_depth)
# true

hasproperty(a, :max_depth)
# true

# Method 8: Get all field values as a tuple
[getfield(a, field) for field in fieldnames(typeof(a))]

# Method 9: Convert to NamedTuple for easy inspection
@btime NamedTuple{fieldnames(typeof(a))}(getfield.((a,), fieldnames(typeof(a))))
# 3.453 μs (14 allocations: 832 bytes)

function show_params(m::MLJ.Model)
    println("Fields of $(typeof(m)):")
    NamedTuple{fieldnames(typeof(m))}(getfield.((m,), fieldnames(typeof(m))))
end
show_params(obj::MLJModel) = show_params(obj())

show_params(decisiontreeclassifier)

@btime show_params(decisiontreeclassifier)
# 7.977 μs (22 allocations: 952 bytes)