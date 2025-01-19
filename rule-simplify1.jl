using AbstractTrees
using DataFrames
using ScikitLearn
using JSON
using Statistics
using SpecialFunctions

# Abstract base type for rule extractors
abstract type BaseRuleExtractor end

# Core data structures
struct Condition
    feature::Int
    operator::Function
    threshold::Float64
    feature_name::Union{String,Nothing}
end

struct Rule
    conditions::Set{Condition}
    class_dist::Vector{Float64}
    ens_class_dist::Vector{Float64}
    logit_score::Union{Float64,Nothing}
    y::Vector{Int}
    y_class_index::Int
    n_samples::Int
    classes::Vector{Int}
    weight::Union{Float64,Nothing}
end

struct RuleSet
    rules::Vector{Rule}
    condition_map::Dict{Int,Condition}
    classes::Vector{Int}
end

# Base methods for rule extraction
function get_base_ruleset(extractor::BaseRuleExtractor, tree_dict::Dict, 
                         class_index=nothing, condition_map=Dict())
    rules = recursive_extraction(extractor, tree_dict, 0, class_index, Set(), condition_map)
    return RuleSet(rules, condition_map, extractor.classes)
end

function recursive_extraction(extractor::BaseRuleExtractor, tree_dict::Dict,
                            node_index::Int, class_index::Union{Int,Nothing},
                            condition_set::Set, condition_map::Dict)
    rules = Rule[]
    
    children_left = tree_dict["children_left"]
    children_right = tree_dict["children_right"]
    features = tree_dict["feature"] 
    thresholds = tree_dict["threshold"]
    
    # Leaf node case
    if children_left[node_index+1] == children_right[node_index+1]
        push!(rules, create_new_rule(extractor, node_index, tree_dict, 
                                   condition_set, nothing, nothing, 
                                   nothing, class_index))
        return rules
    end
    
    # Internal node case - create conditions and recurse
    feature = features[node_index+1]
    threshold = thresholds[node_index+1]
    
    # Create left and right conditions
    op_left, op_right = get_split_operators(extractor)
    
    left_condition = Condition(feature, op_left, threshold, 
                             get_feature_name(extractor, feature))
    right_condition = Condition(feature, op_right, threshold,
                              get_feature_name(extractor, feature))
    
    # Add conditions to maps and sets
    condition_map[hash(left_condition)] = left_condition
    condition_map[hash(right_condition)] = right_condition
    
    # Recurse left
    left_set = copy(condition_set)
    push!(left_set, (hash(left_condition), left_condition))
    append!(rules, recursive_extraction(extractor, tree_dict,
                                     children_left[node_index+1],
                                     class_index, left_set, condition_map))
    
    # Recurse right  
    right_set = copy(condition_set)
    push!(right_set, (hash(right_condition), right_condition))
    append!(rules, recursive_extraction(extractor, tree_dict,
                                     children_right[node_index+1], 
                                     class_index, right_set, condition_map))
    
    return rules
end

# Concrete implementations for different tree types
struct DecisionTreeRuleExtractor <: BaseRuleExtractor
    ensemble
    column_names::Vector{String}
    classes::Vector{Int}
    X::Matrix
    y::Vector
    float_threshold::Float64
end

struct RandomForestRuleExtractor <: BaseRuleExtractor 
    ensemble
    column_names::Vector{String}
    classes::Vector{Int}
    X::Matrix
    y::Vector
    float_threshold::Float64
end

# Add more concrete implementations as needed

# Helper functions
function get_split_operators(::BaseRuleExtractor)
    return (≤, >)  # Default operators
end

function get_feature_name(extractor::BaseRuleExtractor, feature::Int)
    isnothing(extractor.column_names) ? nothing : extractor.column_names[feature+1]
end

# Factory function
function create_rule_extractor(ensemble, column_names, classes, X, y, float_threshold)
    if typeof(ensemble) <: DecisionTreeClassifier
        return DecisionTreeRuleExtractor(ensemble, column_names, classes, X, y, float_threshold)
    elseif typeof(ensemble) <: RandomForestClassifier
        return RandomForestRuleExtractor(ensemble, column_names, classes, X, y, float_threshold)
    else
        error("Unsupported ensemble type")
    end
end
