# ---------------------------------------------------------------------------- #
#                                      debug                                   #
# ---------------------------------------------------------------------------- #
using SoleModels: RuleExtractor
const Optional{T}   = Union{T, Nothing}
const OptFloat64 = Optional{Float64}
using SolePostHoc.RuleExtraction: intrees

using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(;shuffle=true),
    rng=Xoshiro(1),
)
solemc = train_test(dsc)

model = solemc.sole[1]
i = 1
test = get_test(dsc.pidxs[i])
X, y = get_X(dsc)[test, :], get_y(dsc)[test]

# ---------------------------------------------------------------------------- #
#                                      utils                                   #
# ---------------------------------------------------------------------------- #
function _get_rule_extractor_docstring(ruleextractorname::String, method)
  return """Extract rules from a symbolic model using [`$(string(method))`](ref).""" *
         "\n\n" *
         """See also [`modalextractrules`](@ref), [`RuleExtractor`](@ref)."""
end

# ---------------------------------------------------------------------------- #
#                            InTreesRuleExtractor                              #
# ---------------------------------------------------------------------------- #
"""$(_get_rule_extractor_docstring("InTreesRuleExtractor", intrees))"""
struct InTreesRuleExtractor <: RuleExtractor
    prune_rules             :: Bool
    pruning_s               :: OptFloat64
    pruning_decay_threshold :: OptFloat64
    rule_selection_method   :: Symbol
    rule_complexity_metric  :: Symbol
    max_rules               :: Int
    min_coverage            :: OptFloat64
    rng                     :: AbstractRNG

    function InTreesRuleExtractor(;
        prune_rules             :: Bool        = true,
        pruning_s               :: OptFloat64  = 1.0e-6,
        pruning_decay_threshold :: OptFloat64  = 0.05,
        rule_selection_method   :: Symbol      = :CBC,
        rule_complexity_metric  :: Symbol      = :natoms,
        max_rules               :: Int         = -1,
        min_coverage            :: OptFloat64  = 0.01,
        rng                     :: AbstractRNG = TaskLocalRNG()
    )
        return new(
            prune_rules,
            pruning_s,
            pruning_decay_threshold,
            rule_selection_method,
            rule_complexity_metric,
            max_rules,
            min_coverage,
            rng
        )
    end
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
get_prune_rules(e::InTreesRuleExtractor)             = e.prune_rules
get_pruning_s(e::InTreesRuleExtractor)               = e.pruning_s
get_pruning_decay_threshold(e::InTreesRuleExtractor) = e.pruning_decay_threshold
get_rule_selection_method(e::InTreesRuleExtractor)   = e.rule_selection_method
get_rule_complexity_metric(e::InTreesRuleExtractor)  = e.rule_complexity_metric
get_max_rules(e::InTreesRuleExtractor)               = e.max_rules
get_min_coverage(e::InTreesRuleExtractor)            = e.min_coverage
get_rng(e::InTreesRuleExtractor)                     = e.rng

function Base.iterate(e::InTreesRuleExtractor, state=1)
    fields = fieldnames(typeof(e))
    if state > length(fields)
        return nothing
    end
    field = fields[state]
    return (field => getfield(e, field)), state + 1
end

function Base.show(io::IO, info::InTreesRuleExtractor)
    println(io, "InTreesRuleExtractor:")
    for field in fieldnames(InTreesRuleExtractor)
        value = getfield(info, field)
        println(io, "  ", rpad(String(field) * ":", 25), value)
    end
end

function extractrules(e::InTreesRuleExtractor, m, args...; kwargs...)
  dl = intrees(m, e, args...; kwargs...)
#   ll = listrules(dl, use_shortforms=false) # decision list to list of rules
#   rules_obj = convert_classification_rules(dl, ll)
#   dsintrees = DecisionSet(rules_obj)
#   return dsintrees
end

# ---------------------------------------------------------------------------- #
#                                    intrees                                   #
# ---------------------------------------------------------------------------- #
function intrees(
    model::AbstractModel,
    extractor::InTreesRuleExtractor,
    X::AbstractDataFrame,
    y::AbstractVector{<:Label};
    silent::Bool = true,
    return_info::Bool = false,

    # # InTreesRuleExtractor
    # prune_rules::Bool = true,
    # pruning_s::OptFloat64 = nothing,
    # pruning_decay_threshold::OptFloat64 = nothing,
    # rule_selection_method::Symbol = :CBC,
    # rule_complexity_metric::Symbol = :natoms,
    # max_rules::Int = -1,
    # min_coverage::OptFloat64 = nothing,
    # rng::AbstractRNG = MersenneTwister(1)
)
    # if !(X isa AbstractInterpretationSet)
    #     X = SoleData.scalarlogiset(X; allow_propositional = true)
    # end

    """
        cfs()

    Prunes redundant or irrelevant conjuncts of the antecedent of the input rule cascade
    considering the error metric

    See also
    [`Rule`](@ref),
    [`rulemetrics`](@ref).
    """
    function cfs(
        X,
        y::AbstractVector{<:Label},
    )
        entropyd(_x) = ComplexityMeasures.entropy(probabilities(_x))
        midd(_x, _y) = -entropyd(collect(zip(_x, _y)))+entropyd(_x)+entropyd(_y)
        information_gain(f1, f2) = entropyd(f1) - (entropyd(f1) - midd(f1, f2))
        su(f1, f2) = (2.0 * information_gain(f1, f2) / (entropyd(f1) + entropyd(f2)))

        function merit_calculation(X, y::AbstractVector{<:Label})
            n_samples, n_features = size(X)
            rff = 0
            rcf = 0

            for i in collect(1:n_features)
                fi = X[:, i]
                rcf += su(fi, y)  # su is the symmetrical uncertainty of fi and y
                for j in collect(1:n_features)
                    if j > i
                        fj = X[:, j]
                        rff += su(fi, fj)
                    end
                end
            end

            rff *= 2
            merits = rcf / sqrt(n_features + rff)
        end

        n_samples, n_features = size(X)
        F = [] # vector of returned indices samples
        M = [] # vector which stores the merit values

        while true
            merit = -100000000000
            idx = -1

            for i in collect(1:n_features)
                if i ∉ F
                    append!(F,i)
                    idxs_column = F[findall(F .> 0)]
                    t = merit_calculation(X[:,idxs_column],y)

                    if t > merit
                        merit = t
                        idx = i
                    end

                    pop!(F)
                end
            end

            append!(F,idx)
            append!(M,merit)

            # (length(M) > 5) &&
            #     (M[length(M)] <= M[length(M)-1]) &&
            #         (M[length(M)-1] <= M[length(M)-2]) &&
            #             (M[length(M)-2] <= M[length(M)-3]) &&
            #                 (M[length(M)-3] <= M[length(M)-4]) && break

            # Check if merit values are decreasing for the last 5 iterations
            if length(M) > 5 && all(diff(M[end-4:end]) .<= 0)
                break
            end
        end

        valid_idxs = findall(F .> 0)
        # @show F
        # @show valid_idxs
        return F[valid_idxs]
    end

    function starterruleset(model; kwargs...)
        unique(reduce(vcat, [listrules(subm; kwargs...) for subm in SoleModels.models(model)]))
        # TODO maybe also sort?
    end

    if !haslistrules(model)
        model = solemodel(model)
    end

    info = (;)

    ########################################################################################
    # Extract rules from each tree, obtain full ruleset
    ########################################################################################
    silent || println("Extracting starting rules...")
    listrules_kwargs = (;
        use_shortforms=true,
        # flip_atoms = true,
        normalize = true,
        # normalize_kwargs = (; forced_negation_removal = true, reduce_negations = true, allow_atom_flipping = true, rotate_commutatives = false)
    )
    ruleset = isensemble(model) ? starterruleset(model; listrules_kwargs...) : listrules(model; listrules_kwargs...)

    ########################################################################################
    # Prune rules with respect to a dataset
    ########################################################################################
    if prune_rules
        silent || println("Pruning $(length(ruleset)) rules...")
        if return_info
            info = merge(info, (; unpruned_ruleset = ruleset))
        end
        ruleset = @time begin
            afterpruningruleset = Vector{Rule}(undef, length(ruleset))
            Threads.@threads for (i,r) in collect(enumerate(ruleset))
                if r.antecedent isa SoleLogics.BooleanTruth
                    # this case happens with XgBoost: the rule is a simply BooleanTruth
                    # TODO Marco, is this the correct way to handle this case?
                    afterpruningruleset[i] = r
                else
                    afterpruningruleset[i] = intrees_prunerule(r, X, y; pruning_s, pruning_decay_threshold)
                end
            end
            afterpruningruleset
        end
    end

    ########################################################################################
    # Rule selection to obtain the best rules
    ########################################################################################
    silent || println("Selecting via $(string(rule_selection_method)) from a pool of $(length(ruleset)) rules...")
    ruleset = @time begin
        if return_info
            info = merge(info, (; unselected_ruleset = ruleset))
        end
        if rule_selection_method == :CBC
            matrixrulemetrics = Matrix{Float64}(undef,length(ruleset),3)
            afterselectionruleset = Vector{BitVector}(undef, length(ruleset))
            Threads.@threads for (i,rule) in collect(enumerate(ruleset))
                eval_result = rulemetrics(rule, X, y)
                afterselectionruleset[i] = eval_result[:checkmask,]
                matrixrulemetrics[i,1] = eval_result[:coverage]
                matrixrulemetrics[i,2] = eval_result[:error]
                matrixrulemetrics[i,3] = eval_result[rule_complexity_metric]
            end
            #M = hcat([evaluaterule(rule, X, y)[:checkmask,] for rule in ruleset]...)
            M = hcat(afterselectionruleset...)
            
            #best_idxs = findcorrelation(Statistics.cor(M); threshold = accuracy_rule_selection) using Statistics
            #best_idxs = cfs(M,y)
            
            #coefReg = 0.95 .- (0.01*matrixrulemetrics[:,3]/max(matrixrulemetrics[:,3]...))
            #@show coefReg
            rf = DT.build_forest(y,M,2,50,0.7,-1; rng=rng)
            importances = begin
                #importance = impurity_importance(rf, coefReg)
                importance = DT.impurity_importance(rf)
                importance/max(importance...)
            end
            # @show importances
            best_idxs = begin
                selected_features = findall(importances .> 0.01)
                # @show selected_features
                ruleSetPrunedRRF = hcat(matrixrulemetrics[selected_features,:],importances[selected_features],selected_features)
                finalmatrix = sortslices(ruleSetPrunedRRF, dims=1, by=x->(x[4],x[2],x[3]), rev=true)
                
                # Get all selected rules indices or limit if max_rules is specified
                if max_rules > 0
                    best_idxs = Int.(finalmatrix[1:min(max_rules, size(finalmatrix, 1)), 5])
                else
                    best_idxs = Int.(finalmatrix[:,5])
                end
            end
            # @show best_idxs
            ruleset[best_idxs]
        else
            error("Unexpected rule selection method specified: $(rule_selection_method)")
        end
    end
    silent || println("# rules selected: $(length(ruleset)).")
    
    ########################################################################################
    # Construct a rule-based model from the set of best rules
    ########################################################################################
    silent || println("Applying STEL...")
    
    dl = STEL(ruleset, X, y; max_rules, min_coverage, rule_complexity_metric, rng, silent)

    if return_info
        return dl, info
    else
        return dl
    end
end