module RuleHeuristics

using BitArrays
using LinearAlgebra

struct RuleHeuristics
    X::Matrix{Float64}
    y::Vector{Int}
    classes_::Vector{Int}
    n_classes::Int
    condition_map::Dict{Int, Any}
    cov_threshold::Float64
    conf_threshold::Float64
    bitarray_backend::String
    bitarray_
    training_bit_sets
    _cond_cov_dict
    training_heuristics_dict
    ones
    zeros

    function RuleHeuristics(X, y, classes_, condition_map, cov_threshold=0.0, conf_threshold=0.5, bitarray_backend="python-int")
        n_classes = length(classes_)
        bitarray_ = bitarray_backend == "python-int" ? PythonIntArray(size(X, 1), classes_) : BitArray(size(X, 1), classes_)
        ones = bitarray_.generate_ones()
        zeros = bitarray_.generate_zeros()
        new(X, y, classes_, n_classes, condition_map, cov_threshold, conf_threshold, bitarray_backend, bitarray_, nothing, nothing, nothing, ones, zeros)
    end
end

function get_conditions_heuristics(rh::RuleHeuristics, conditions, not_cov_mask=nothing)
    empty_list = zeros(Float64, rh.n_classes)
    heuristics_dict = Dict(
        "cov_set" => [rh.zeros for _ in 1:(rh.n_classes + 1)],
        "cov" => 0.0,
        "cov_count" => 0.0,
        "class_cov_count" => empty_list,
        "conf" => empty_list,
        "supp" => empty_list
    )

    if isempty(conditions)
        return get_training_heuristics_dict(rh, not_cov_mask), not_cov_mask
    end

    b_array_conds = [reduce(&, [rh._cond_cov_dict[i][cond[1]] for cond in conditions]) for i in 1:rh.n_classes]
    push!(b_array_conds, reduce(|, b_array_conds))

    cov_count = rh.bitarray_.get_number_ones(b_array_conds[end])

    if cov_count == 0
        return heuristics_dict, not_cov_mask
    end

    class_cov_count = [rh.bitarray_.get_number_ones(b_array_conds[i]) for i in 1:rh.n_classes]
    coverage = cov_count / size(rh.X, 1)
    heuristics_dict["cov_set"] = b_array_conds
    heuristics_dict["cov"] = coverage
    heuristics_dict["cov_count"] = cov_count
    heuristics_dict["class_cov_count"] = class_cov_count
    heuristics_dict["conf"] = [class_count / cov_count for class_count in class_cov_count]
    heuristics_dict["supp"] = [class_count / size(rh.X, 1) for class_count in class_cov_count]

    return heuristics_dict, not_cov_mask
end

function compute_rule_heuristics(rh::RuleHeuristics, ruleset, not_cov_mask=nothing, sequential_covering=false, recompute=false)
    if recompute
        for rule in ruleset
            heuristics_dict, _ = get_conditions_heuristics(rh, rule.A)
            rule.set_heuristics(heuristics_dict)
        end
        return
    end

    not_cov_mask = not_cov_mask === nothing ? rh.ones : not_cov_mask

    if sequential_covering
        accurate_rules = []
        local_not_cov_samples = not_cov_mask
        for rule in ruleset
            result, not_cov_samples_with_rule = rule_is_accurate(rh, rule, local_not_cov_samples)
            if result
                push!(accurate_rules, rule)
                local_not_cov_samples = not_cov_samples_with_rule
            end
        end
        ruleset.rules[:] = accurate_rules
    else
        for rule in ruleset
            set_rule_heuristics(rh, rule, not_cov_mask)
        end
    end
end

function _compute_training_bit_sets(rh::RuleHeuristics)
    training_bit_set = [rh.bitarray_.get_array(rh.y .== rh.classes_[i]) for i in 1:rh.n_classes]
    push!(training_bit_set, reduce(|, training_bit_set))
    return training_bit_set
end

function _compute_condition_bit_sets(rh::RuleHeuristics)
    cond_cov_dict = [Dict{Int, Any}() for _ in 1:(rh.n_classes + 1)]
    for (cond_id, cond) in rh.condition_map
        cond_coverage_bitarray = rh.bitarray_.get_array(cond.satisfies_array(rh.X))
        for i in 1:rh.n_classes
            cond_cov_dict[i][cond_id] = cond_coverage_bitarray & rh.training_bit_sets[i]
        end
        cond_cov_dict[end][cond_id] = cond_coverage_bitarray
    end
    return cond_cov_dict
end

function initialize_sets(rh::RuleHeuristics)
    rh.training_bit_sets = _compute_training_bit_sets(rh)
    rh._cond_cov_dict = _compute_condition_bit_sets(rh)
end

function rule_is_accurate(rh::RuleHeuristics, rule, not_cov_samples)
    if rh.bitarray_.get_number_ones(not_cov_samples) == 0
        return false, not_cov_samples
    end

    local_not_cov_samples = set_rule_heuristics(rh, rule, not_cov_samples)

    if rule.conf > rh.conf_threshold && rule.cov > rh.cov_threshold
        return true, local_not_cov_samples
    else
        return false, not_cov_samples
    end
end

function create_empty_heuristics_dict(rh::RuleHeuristics)
    empty_list = zeros(Float64, rh.n_classes)
    return Dict(
        "cov_set" => [rh.zeros for _ in 1:(rh.n_classes + 1)],
        "cov" => 0.0,
        "cov_count" => 0.0,
        "class_cov_count" => empty_list,
        "conf" => empty_list,
        "supp" => empty_list
    )
end

function get_training_heuristics_dict(rh::RuleHeuristics, not_cov_mask=nothing)
    if rh.training_heuristics_dict === nothing
        cov_count = rh.bitarray_.get_number_ones(rh.training_bit_sets[end])
        class_cov_count = [rh.bitarray_.get_number_ones(rh.training_bit_sets[i]) for i in 1:rh.n_classes]
        coverage = cov_count / size(rh.X, 1)
        train_heur_dict = Dict(
            "cov_set" => rh.training_bit_sets,
            "cov" => coverage,
            "cov_count" => cov_count,
            "class_cov_count" => class_cov_count,
            "conf" => [class_count / cov_count for class_count in class_cov_count],
            "supp" => [class_count / size(rh.X, 1) for class_count in class_cov_count]
        )
        rh.training_heuristics_dict = train_heur_dict
    end

    if not_cov_mask === nothing
        return rh.training_heuristics_dict
    else
        if rh.bitarray_.get_number_ones(not_cov_mask) == 0
            empty_list = zeros(Float64, rh.n_classes)
            return Dict(
                "cov_set" => [rh.zeros for _ in 1:(rh.n_classes + 1)],
                "cov" => 0.0,
                "cov_count" => 0.0,
                "class_cov_count" => empty_list,
                "conf" => empty_list,
                "supp" => empty_list
            )
        end
        masked_training_heuristics = [b_array_measure & not_cov_mask for b_array_measure in rh.training_bit_sets]
        cov_count = rh.bitarray_.get_number_ones(masked_training_heuristics[end])
        class_cov_count = [rh.bitarray_.get_number_ones(masked_training_heuristics[i]) for i in 1:rh.n_classes]
        coverage = cov_count / size(rh.X, 1)

        return Dict(
            "cov_set" => masked_training_heuristics,
            "cov" => coverage,
            "cov_count" => cov_count,
            "class_cov_count" => class_cov_count,
            "conf" => [class_count / cov_count for class_count in class_cov_count],
            "supp" => [class_count / size(rh.X, 1) for class_count in class_cov_count]
        )
    end
end

function combine_heuristics(rh::RuleHeuristics, heuristics1, heuristics2)
    cov_set = [rh.zeros for _ in 1:(rh.n_classes + 1)]
    for i in 1:(rh.n_classes + 1)
        cov_set[i] = heuristics1["cov_set"][i] & heuristics2["cov_set"][i]
    end
    cov_count = rh.bitarray_.get_number_ones(cov_set[end])

    class_cov_count = [rh.bitarray_.get_number_ones(cov_set[i]) for i in 1:rh.n_classes]

    coverage = cov_count / size(rh.X, 1)
    if coverage == 0
        empty_list = zeros(Float64, rh.n_classes)
        return Dict(
            "cov_set" => cov_set,
            "cov" => 0.0,
            "cov_count" => 0.0,
            "class_cov_count" => class_cov_count,
            "conf" => empty_list,
            "supp" => empty_list
        )
    end
    return Dict(
        "cov_set" => cov_set,
        "cov" => coverage,
        "cov_count" => cov_count,
        "class_cov_count" => class_cov_count,
        "conf" => [class_count / cov_count for class_count in class_cov_count],
        "supp" => [class_count / size(rh.X, 1) for class_count in class_cov_count]
    )
end

function set_rule_heuristics(rh::RuleHeuristics, rule, mask)
    mask_cov_set = [cov_set & mask for cov_set in rule.heuristics_dict["cov_set"]]

    cov_count = rh.bitarray_.get_number_ones(mask_cov_set[end])

    if cov_count == 0
        rule.conf = 0.0
        rule.supp = 0.0
        rule.cov = 0.0
        return rh.bitarray_.get_complement(mask_cov_set[end], rh.ones) & mask
    else
        class_cov_count = [rh.bitarray_.get_number_ones(mask_cov_set[i]) for i in 1:rh.n_classes]

        coverage = cov_count / size(rh.X, 1)

        rule.conf = class_cov_count[rule.class_index] / cov_count
        rule.supp = class_cov_count[rule.class_index] / size(rh.X, 1)
        rule.cov = coverage
        rule.n_samples = class_cov_count

        return rh.bitarray_.get_complement(mask_cov_set[end], rh.ones) & mask
    end
end

end # module