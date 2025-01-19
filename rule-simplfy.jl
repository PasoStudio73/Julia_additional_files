module RuleExtraction

using JSON
using LinearAlgebra
using Statistics

abstract type AbstractRuleExtractor end

struct Rule
    condition_set::Set
    class_dist::Vector{Float64}
    ens_class_dist::Vector{Float64}
    logit_score::Union{Nothing, Float64}
    y::Vector{Int}
    y_class_index::Int
    n_samples::Int
    classes::Vector{Int}
    weight::Union{Nothing, Float64}
end

struct RuleSet
    rules::Vector{Rule}
    condition_map::Dict{Int, Any}
    classes::Vector{Int}
end

struct Condition
    feature::Int
    operator::Function
    threshold::Float64
    att_name::Union{Nothing, String}
end

mutable struct BaseRuleExtractor <: AbstractRuleExtractor
    _ensemble
    _column_names::Union{Nothing, Vector{String}}
    classes_::Vector{Int}
    X::Matrix{Float64}
    y::Vector{Int}
    float_threshold::Float64
    class_ratio::Float64

    function BaseRuleExtractor(_ensemble, _column_names, classes_, X, y, float_threshold)
        _, counts = unique(y, return_counts=true)
        class_ratio = minimum(counts) / maximum(counts)
        new(_ensemble, _column_names, classes_, X, y, float_threshold, class_ratio)
    end
end

function get_tree_dict(base_tree, n_nodes=0)
    return Dict(
        "children_left" => base_tree.tree_.children_left,
        "children_right" => base_tree.tree_.children_right,
        "feature" => base_tree.tree_.feature,
        "threshold" => base_tree.tree_.threshold,
        "value" => base_tree.tree_.value,
        "n_samples" => base_tree.tree_.weighted_n_node_samples,
        "n_nodes" => base_tree.tree_.node_count
    )
end

function get_split_operators()
    return (<=, >)
end

function recursive_extraction(extractor::BaseRuleExtractor, tree_dict, tree_index=0, class_index=nothing, node_index=0, condition_map=nothing, condition_set=nothing)
    condition_map = condition_map === nothing ? Dict() : condition_map
    condition_set = condition_set === nothing ? Set() : condition_set
    rules = Vector{Rule}()
    children_left = tree_dict["children_left"]
    children_right = tree_dict["children_right"]
    feature = tree_dict["feature"]
    threshold = tree_dict["threshold"]

    if children_left[node_index] == children_right[node_index]
        weights = nothing
        logit_score = nothing
        new_rule = create_new_rule(extractor, node_index, tree_dict, condition_set, logit_score, weights, tree_index, class_index)
        push!(rules, new_rule)
    else
        att_name = extractor._column_names !== nothing ? extractor._column_names[feature[node_index]] : nothing
        condition_set_left = copy(condition_set)
        op_left, op_right = get_split_operators()

        split_value = threshold[node_index]
        if abs(split_value) < extractor.float_threshold
            split_value = copysign(extractor.float_threshold, split_value)
        end
        new_condition_left = Condition(feature[node_index], op_left, split_value, att_name)
        condition_map[hash(new_condition_left)] = new_condition_left
        push!(condition_set_left, (hash(new_condition_left), new_condition_left))
        left_rules = recursive_extraction(extractor, tree_dict, tree_index, class_index, children_left[node_index], condition_map, condition_set_left)
        append!(rules, left_rules)

        condition_set_right = copy(condition_set)
        new_condition_right = Condition(feature[node_index], op_right, split_value, att_name)
        condition_map[hash(new_condition_right)] = new_condition_right
        push!(condition_set_right, (hash(new_condition_right), new_condition_right))
        right_rules = recursive_extraction(extractor, tree_dict, tree_index, class_index, children_right[node_index], condition_map, condition_set_right)
        append!(rules, right_rules)
    end
    return rules
end

function create_new_rule(extractor::BaseRuleExtractor, node_index, tree_dict, condition_set=nothing, logit_score=nothing, weights=nothing, tree_index=nothing, class_index=nothing)
    condition_set = condition_set === nothing ? Set() : condition_set
    value = tree_dict["value"]
    n_samples = tree_dict["n_samples"]

    weight = weights !== nothing ? weights[tree_index] : nothing
    class_dist = (value[node_index] / sum(value[node_index]))[:]
    y_class_index = argmax(class_dist)
    y = [extractor.classes_[y_class_index]]

    return Rule(frozenset(condition_set), class_dist, class_dist, logit_score, y, y_class_index, n_samples[node_index], extractor.classes_, weight)
end

end # module