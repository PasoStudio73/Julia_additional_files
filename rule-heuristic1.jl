using BitArrays
using Statistics

"""
RuleHeuristics handles computation of rule metrics using efficient bit operations.
"""
struct RuleHeuristics
    X::Matrix{Float64}
    y::Vector{Int}
    classes::Vector{Int}
    n_classes::Int
    condition_map::Dict{Int,Condition}
    cov_threshold::Float64
    conf_threshold::Float64
    training_bit_sets::Vector{BitVector}
    cond_cov_dict::Vector{Dict{Int,BitVector}}
    ones::BitVector
    zeros::BitVector
    
    function RuleHeuristics(X::Matrix{Float64}, y::Vector{Int}, 
                          classes::Vector{Int}, condition_map::Dict{Int,Condition},
                          cov_threshold::Float64=0.0, conf_threshold::Float64=0.5)
        n_classes = length(classes)
        n_samples = size(X, 1)
        
        # Initialize bit arrays
        ones = trues(n_samples)
        zeros = falses(n_samples)
        
        # Create new instance
        rh = new(X, y, classes, n_classes, condition_map,
                cov_threshold, conf_threshold,
                Vector{BitVector}(), Vector{Dict{Int,BitVector}}(),
                ones, zeros)
                
        # Initialize bit sets
        initialize_sets!(rh)
        
        return rh
    end
end

"""
Initialize bit sets for training data and conditions.
"""
function initialize_sets!(rh::RuleHeuristics)
    # Training bit sets
    rh.training_bit_sets = [BitVector(rh.y .== class) for class in rh.classes]
    push!(rh.training_bit_sets, reduce(.|, rh.training_bit_sets))
    
    # Condition bit sets
    rh.cond_cov_dict = [Dict{Int,BitVector}() for _ in 1:(rh.n_classes+1)]
    
    for (cond_id, cond) in rh.condition_map
        # Compute condition coverage
        cond_coverage = BitVector(satisfies.(Ref(cond), eachrow(rh.X)))
        
        # Store class-specific coverages
        for i in 1:rh.n_classes
            rh.cond_cov_dict[i][cond_id] = cond_coverage .& rh.training_bit_sets[i]
        end
        rh.cond_cov_dict[end][cond_id] = cond_coverage
    end
end

"""
Compute heuristics for a set of conditions.
"""
function get_conditions_heuristics(rh::RuleHeuristics, 
                                 conditions::Set, 
                                 not_cov_mask::Union{BitVector,Nothing}=nothing)
    
    empty_metrics = zeros(Float64, rh.n_classes)
    heuristics = Dict(
        :cov_set => [rh.zeros for _ in 1:(rh.n_classes+1)],
        :cov => 0.0,
        :cov_count => 0.0,
        :class_cov_count => copy(empty_metrics),
        :conf => copy(empty_metrics),
        :supp => copy(empty_metrics)
    )
    
    isempty(conditions) && return heuristics, not_cov_mask
    
    # Compute coverage bit arrays
    b_arrays = [reduce(&, [rh.cond_cov_dict[i][cond[1]] for cond in conditions])
                for i in 1:rh.n_classes]
    push!(b_arrays, reduce(|, b_arrays))
    
    cov_count = count(b_arrays[end])
    cov_count == 0 && return heuristics, not_cov_mask
    
    # Compute metrics
    class_cov_counts = count.(b_arrays[1:end-1])
    coverage = cov_count / size(rh.X, 1)
    
    return Dict(
        :cov_set => b_arrays,
        :cov => coverage,
        :cov_count => cov_count,
        :class_cov_count => class_cov_counts,
        :conf => class_cov_counts ./ cov_count,
        :supp => class_cov_counts ./ size(rh.X, 1)
    ), not_cov_mask
end

"""
Compute rule accuracy and update coverage mask.
"""
function rule_is_accurate(rh::RuleHeuristics, rule::Rule, not_cov_samples::BitVector)
    count(not_cov_samples) == 0 && return false, not_cov_samples
    
    local_not_cov = set_rule_heuristics!(rh, rule, not_cov_samples)
    
    if rule.conf > rh.conf_threshold && rule.cov > rh.cov_threshold
        return true, local_not_cov
    end
    return false, not_cov_samples
end

"""
Combine heuristics from two rules.
"""
function combine_heuristics(rh::RuleHeuristics, h1::Dict, h2::Dict)
    cov_sets = [h1[:cov_set][i] .& h2[:cov_set][i] 
                for i in 1:(rh.n_classes+1)]
    
    cov_count = count(cov_sets[end])
    class_counts = count.(cov_sets[1:end-1])
    coverage = cov_count / size(rh.X, 1)
    
    coverage == 0 && return create_empty_heuristics(rh)
    
    return Dict(
        :cov_set => cov_sets,
        :cov => coverage,
        :cov_count => cov_count,
        :class_cov_count => class_counts,
        :conf => class_counts ./ cov_count,
        :supp => class_counts ./ size(rh.X, 1)
    )
end
