using SoleXplorer
using SoleModels
using MLJ
using DataFrames, Random
using Test, BenchmarkTools
import DecisionTree as DT
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

modelc = symbolic_analysis(Xc, yc)
ds = modelc.ds

# ---------------------------------------------------------------------------- #
struct PasoConstantModel{O} <: LeafModel{O}
    outcome::O
    info::Base.RefValue{<:NamedTuple}

    function PasoConstantModel{O}(
        outcome::O2,
        info::Base.RefValue{<:NamedTuple}
    ) where {O,O2}
        new{O}(convert(O, outcome), info)
    end

    function PasoConstantModel(
        outcome::O,
        info::Base.RefValue{<:NamedTuple}
    ) where {O}
        PasoConstantModel{O}(outcome, info)
    end

    function PasoConstantModel{O}(m::PasoConstantModel) where {O}
        PasoConstantModel{O}(m.outcome, m.info)
    end

    function PasoConstantModel(m::PasoConstantModel)
        PasoConstantModel(m.outcome, m.info)
    end
end

outcome(m::PasoConstantModel) = m.outcome
leafmodelname(m::PasoConstantModel) = string(outcome(m))
iscomplete(::PasoConstantModel) = true

# ---------------------------------------------------------------------------- #
struct PasoBranch{O} <: AbstractModel{O}
    antecedent::Formula
    posconsequent::M where {M<:AbstractModel{<:O}}
    negconsequent::M where {M<:AbstractModel{<:O}}
    info::Base.RefValue{<:NamedTuple}

    function PasoBranch{O}(
        antecedent::Formula,
        posconsequent::Any,
        negconsequent::Any,
        info::Base.RefValue{<:NamedTuple}
    ) where {O}
        A = typeof(antecedent)
        posconsequent = wrap(posconsequent)
        negconsequent = wrap(negconsequent)
        new{O}(antecedent, posconsequent, negconsequent, info)
    end

    function PasoBranch(
        antecedent::Formula,
        posconsequent::Any,
        negconsequent::Any,
        info::Base.RefValue{<:NamedTuple}
    )
        A = typeof(antecedent)
        posconsequent = wrap(posconsequent)
        negconsequent = wrap(negconsequent)
        O = Union{outcometype(posconsequent),outcometype(negconsequent)}
        PasoBranch{O}(antecedent, posconsequent, negconsequent, info)
    end

    function PasoBranch(
        antecedent::Formula,
        (posconsequent, negconsequent)::Tuple{Any,Any},
        info::Base.RefValue{<:NamedTuple}
    )
        PasoBranch(antecedent, posconsequent, negconsequent, info)
    end
end

antecedent(m::PasoBranch) = m.antecedent
posconsequent(m::PasoBranch) = m.posconsequent
negconsequent(m::PasoBranch) = m.negconsequent
iscomplete(m::PasoBranch) = iscomplete(posconsequent(m)) && iscomplete(negconsequent(m))
immediatesubmodels(m::PasoBranch) = [posconsequent(m), negconsequent(m)]
nimmediatesubmodels(m::PasoBranch) = 2
listimmediaterules(m::PasoBranch{O}) where {O} = [
    Rule{O}(antecedent(m), posconsequent(m)),
    Rule{O}(SoleLogics.NEGATION(antecedent(m)), negconsequent(m)),
]

pasocheckantecedent(
    m::Union{Rule,PasoBranch},
    i::SoleModels.AbstractInterpretation,
    args...;
    kwargs...
) = check(antecedent(m), i, args...; kwargs...)
pasocheckantecedent(
    m::Union{Rule,PasoBranch},
    d::SoleModels.AbstractInterpretationSet,
    i_instance::Integer,
    args...;
    kwargs...
) = check(antecedent(m), d, i_instance, args...; kwargs...)
pasocheckantecedent(
    m::Union{Rule,PasoBranch},
    d::SoleModels.AbstractInterpretationSet,
    args...;
    kwargs...
) = check(antecedent(m), d, args...; kwargs...)

# ---------------------------------------------------------------------------- #
mutable struct PasoDecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable,W<:Union{Nothing,AbstractVector}} <: SoleModels.AbstractDecisionEnsemble{O}
    models::Vector{T}
    aggregation::A
    weights::W
    info::NamedTuple

    function PasoDecisionEnsemble{O}(
        models::AbstractVector{T},
        aggregation::Union{Nothing,Base.Callable},
        weights::Union{Nothing,AbstractVector},
        info::NamedTuple=(;);
        suppress_parity_warning=false,
        parity_func=x->argmax(x)
    ) where {O,T<:AbstractModel}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        if isnothing(aggregation)
            aggregation = function (args...; suppress_parity_warning, kwargs...) SoleModels.bestguess(args...; suppress_parity_warning, parity_func, kwargs...) end
        else
            !suppress_parity_warning || @warn "Unexpected value for suppress_parity_warning: $(suppress_parity_warning)."
        end
        # T = typeof(models)
        W = typeof(weights)
        A = typeof(aggregation)
        new{O,T,A,W}(collect(models), aggregation, weights, info)
    end
    
    function PasoDecisionEnsemble{O}(
        models::AbstractVector;
        kwargs...
    ) where {O}
        info=(;)
        PasoDecisionEnsemble{O}(models, nothing, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        info::NamedTuple;
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        aggregation::Union{Nothing,Base.Callable},
        info::NamedTuple=(;);
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, aggregation, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        weights::AbstractVector,
        info::NamedTuple=(;);
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, weights, info; kwargs...)
    end

    function PasoDecisionEnsemble(
        models::AbstractVector,
        args...; kwargs...
    )
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        O = Union{outcometype.(models)...}
        PasoDecisionEnsemble{O}(models, args...; kwargs...)
    end
end

mutable struct PasoDecisionTree{O} <: AbstractModel{O}
    root::Union{LeafModel{O},PasoBranch{O}}
    info::NamedTuple

    function PasoDecisionTree(
        root::Union{LeafModel{O},PasoBranch{O}},
        info::NamedTuple=(;)
    ) where {O}
        new{O}(root, info)
    end

    function PasoDecisionTree(
        root::Any,
        info::NamedTuple=(;)
    )
        root = wrap(root)
        M = typeof(root)
        O = outcometype(root)
        @assert M <: Union{LeafModel{O},PasoBranch{O}} "" *
            "Cannot instantiate PasoDecisionTree{$(O)}(...) with root of " *
            "type $(typeof(root)). Note that the should be either a LeafModel or a " *
            "PasoBranch. " *
            "$(M) <: $(Union{LeafModel,PasoBranch{<:O}}) should hold."
        new{O}(root, info)
    end

    function PasoDecisionTree(
        antecedent::Formula,
        posconsequent::Any,
        negconsequent::Any,
        info::NamedTuple=(;)
    )
        posconsequent isa PasoDecisionTree && (posconsequent = root(posconsequent))
        negconsequent isa PasoDecisionTree && (negconsequent = root(negconsequent))
        return PasoDecisionTree(PasoBranch(antecedent, posconsequent, negconsequent, Ref(info)), info)
    end
end

# ---------------------------------------------------------------------------- #
function get_featurenames(tree::Union{DT.Ensemble, DT.InfoNode})
    if !hasproperty(tree, :info)
        throw(ArgumentError("Please provide featurenames."))
    end
    return tree.info.featurenames
end
get_classlabels(tree::Union{DT.Ensemble, DT.InfoNode})::Vector{<:SX.Label} = tree.info.classlabels

function get_condition(featid, featval, featurenames)
    test_operator = (<)
    feature = isnothing(featurenames) ? VariableValue(featid) : VariableValue(featid, featurenames[featid])
    return ScalarCondition(feature, test_operator, featval)
end

# ---------------------------------------------------------------------------- #
function pasomodel(
    model          :: DT.Ensemble{T,O};
    featurenames   :: Vector{Symbol}=Symbol[],
    weights        :: Vector{<:Number}=Number[],
    classlabels    :: AbstractVector{<:SoleModels.Label}=SoleModels.Label[],
    keep_condensed :: Bool=false,
    parity_func    :: Base.Callable=x->first(sort(collect(keys(x))))
)::PasoDecisionEnsemble where {T,O}
    isempty(featurenames) && (featurenames = get_featurenames(model))
    info= (
        supporting_predictions=SX.CLabel[],
        supporting_labels=SX.CLabel[],
        featurenames=Symbol[],
        classlabels=Symbol[]
    )

    trees = map(t -> pasomodel(t, Ref(info); featurenames, classlabels), model.trees)

    isnothing(weights) ?
        PasoDecisionEnsemble{O}(trees, info; parity_func) :
        PasoDecisionEnsemble{O}(trees, weights, info; parity_func)
end

function pasomodel(
    tree           :: DT.InfoNode{T,O};
    featurenames   :: Union{Nothing,Vector{Symbol}}=nothing,
)::PasoDecisionTree where {T,O}
    isnothing(featurenames) && (featurenames = get_featurenames(tree))
    classlabels = hasproperty(tree.info, :classlabels) ? get_classlabels(tree) : SX.Label[]
    info= (
        supporting_predictions=SX.CLabel[],
        supporting_labels=SX.CLabel[],
        featurenames=Symbol[],
        classlabels=Symbol[]
    )
    root = pasomodel(tree.node, Ref(info); featurenames, classlabels)

    PasoDecisionTree(root, info)
end

function pasomodel(
    tree         :: DT.Node,
    info         :: Base.RefValue{<:NamedTuple};
    featurenames :: Vector{Symbol},
    classlabels  :: AbstractVector{<:SX.Label}=SX.Label[],
)::PasoBranch
    cond = get_condition(tree.featid, tree.featval, featurenames)
    antecedent = Atom(cond)
    lefttree  = pasomodel(tree.left, info; featurenames, classlabels )
    righttree = pasomodel(tree.right, info; featurenames, classlabels )

    return PasoBranch(antecedent, lefttree, righttree, info)
end

function pasomodel(
    tree         :: DT.Leaf,
    info         :: Base.RefValue{<:NamedTuple};
    featurenames :: Vector{Symbol},
    classlabels  :: AbstractVector{<:SX.Label}=SX.Label[]
)::PasoConstantModel
    prediction = isempty(classlabels) ? tree.majority : classlabels[tree.majority]

    PasoConstantModel(prediction, info)
end

# ---------------------------------------------------------------------------- #
featurenames = MLJ.report(ds.mach).features
solem = pasomodel(MLJ.fitted_params(ds.mach).tree; featurenames);

@btime pasomodel(MLJ.fitted_params(ds.mach).tree; featurenames);
# 5.093 μs (98 allocations: 4.34 KiB)
@btime solemodel(MLJ.fitted_params(ds.mach).tree; featurenames);
# 14.260 μs (139 allocations: 10.94 KiB)

ds = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)
train, test = get_train(ds.pidxs[1]), get_test(ds.pidxs[1])
X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]
MLJ.fit!(ds.mach, rows=train, verbosity=0)
classlabels  = ds.mach.fitresult[2][sortperm((ds.mach).fitresult[3])]
featurenames = MLJ.report(ds.mach).features
@btime pasomodel(MLJ.fitted_params(ds.mach).forest; featurenames);
# 308.047 μs (3322 allocations: 128.24 KiB)
@btime solemodel(MLJ.fitted_params(ds.mach).forest; featurenames);
# 1.287 ms (6452 allocations: 436.05 KiB)

# ---------------------------------------------------------------------------- #
pasoroot(m::PasoDecisionTree) = m.root
models(m::PasoDecisionEnsemble) = m.models

function set_predictions(
    info  :: NamedTuple,
    preds :: AbstractVector{T},
    y     :: AbstractVector{S}
)::NamedTuple where {T,S<:SoleModels.Label}
    merge(info, (supporting_predictions=preds, supporting_labels=y))
end

aggregation(m::PasoDecisionEnsemble) = m.aggregation
weights(m::PasoDecisionEnsemble) = m.weights
# Returns the aggregation function, patched by weights if the model has them.
function weighted_aggregation(m::PasoDecisionEnsemble)
    if isempty(weights(m))
        aggregation(m)
    else
        function (labels; kwargs...)
            aggregation(m)(labels, weights(m); kwargs...)
        end
    end
end

# ---------------------------------------------------------------------------- #
function pasoapply!(
    m::PasoDecisionEnsemble,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    mode = :replace,
    leavesonly = false,
    suppress_parity_warning = false,
    kwargs...
)
    preds = hcat([pasoapply!(subm, d, y; mode, leavesonly, kwargs...) for subm in models(m)]...)
    preds = [
        weighted_aggregation(m)(preds[i,:]; suppress_parity_warning, kwargs...)
        for i in 1:size(preds,1)
    ]

    m.info  = set_predictions(m.info, preds, y)
    return nothing
end

function pasoapply!(
    m::PasoDecisionTree,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    mode = :replace,
    leavesonly = false,
    kwargs...
)
    preds = pasoapply!(pasoroot(m), d, y;
        mode = mode,
        leavesonly = leavesonly,
        kwargs...
    )

    m.info  = set_predictions(m.info, preds, y)
    return nothing
end

function pasoapply!(
    m::PasoBranch,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    mode = :replace,
    leavesonly = false,
    kwargs...
)
    # @assert length(y) == ninstances(d) "$(length(y)) == $(ninstances(d))"
    checkmask = pasocheckantecedent(m, d, check_args...; check_kwargs...)
    preds = Vector{outcometype(m)}(undef,length(checkmask)) ## HACKERATA da non copiare
    @sync begin
        if any(checkmask)
            l = Threads.@spawn pasoapply!(
                posconsequent(m),
                slicedataset(d, checkmask; return_view = true),
                y[checkmask];
                check_args = check_args,
                check_kwargs = check_kwargs,
                # mode = mode,
                leavesonly = leavesonly,
                kwargs...
            )
        end
        ncheckmask = (!).(checkmask)
        if any(ncheckmask)
            r = Threads.@spawn pasoapply!(
                negconsequent(m),
                slicedataset(d, ncheckmask; return_view = true),
                y[ncheckmask];
                check_args = check_args,
                check_kwargs = check_kwargs,
                # mode = mode,
                leavesonly = leavesonly,
                kwargs...
            )
        end
        if any(checkmask)
            preds[checkmask] .= fetch(l)
        end
        if any(ncheckmask)
            preds[ncheckmask] .= fetch(r)
        end
    end
    return preds
end

function pasoapply!(
    m::PasoConstantModel,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    # mode = :replace,
    # leavesonly = false,
    kwargs...
)
    preds = fill(outcome(m), ninstances(d))

    return preds
end

# ---------------------------------------------------------------------------- #
#                        nuova SoleXplorer train_test                          #
# ---------------------------------------------------------------------------- #
# Per poter fare dei benchmark comparativi, preferirei scrivere una nuova funzione train_test
# di Sole, che usa le nuove funzioni

function xplorer_apply(
    ds :: SoleXplorer.DecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    solem        = pasomodel(MLJ.fitted_params(ds.mach).tree; featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    pasoapply!(solem, logiset, y)
    return solem
end

function xplorer_apply(
    ds :: SoleXplorer.PropositionalDataSet{RandomForestClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    classlabels  = ds.mach.fitresult[2][sortperm((ds.mach).fitresult[3])]
    featurenames = MLJ.report(ds.mach).features
    solem        = pasomodel(MLJ.fitted_params(ds.mach).forest; classlabels, featurenames)
    logiset      = scalarlogiset(X, allow_propositional = true)
    pasoapply!(solem, logiset, y)
    return solem
end

function _paso_test(ds::SoleXplorer.EitherDataSet)::SoleXplorer.SModel
    n_folds   = length(ds.pidxs)
    solemodel = Vector{AbstractModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds @views for i in 1:n_folds
        train, test = get_train(ds.pidxs[i]), get_test(ds.pidxs[i])
        X_test, y_test = get_X(ds)[test, :], get_y(ds)[test]

        SoleXplorer.has_xgboost_model(ds) && SoleXplorer.set_watchlist!(ds, i)

        MLJ.fit!(ds.mach, rows=train, verbosity=0)
        solemodel[i] = xplorer_apply(ds, X_test, y_test)
    end

    return SoleXplorer.SModel(ds, solemodel)
end

function paso_test(args...; kwargs...)::SoleXplorer.SModel
    ds = SoleXplorer._setup_dataset(args...; kwargs...)
    _paso_test(ds)
end

paso_test(ds::SoleXplorer.AbstractDataSet)::SoleXplorer.SModel = _paso_test(ds)

# per completare l'opera dobbiamo scrivere i metodi di apply! che accettano PasoDecisionTree e PasoEnsemble

# Verifichiamo il corretto funzionamento
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)
solemc = paso_test(dsc)
model_new = symbolic_analysis(
    dsc, solemc;
    measures=(accuracy, kappa)
)

# Esperimento originale
model_old = symbolic_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

@test model_new.measures.measures_values == model_old.measures.measures_values
# test superato

# Ora diamo un occhiata ai benchmark per vedere se effettivamente siamo migliorati
@btime begin
    setup_dataset(
        Xc, yc;
        model=DecisionTreeClassifier(),
        resample=Holdout(shuffle=true),
            train_ratio=0.7,
            rng=Xoshiro(1),   
    )
    solemc = paso_test(dsc)
    model_new = symbolic_analysis(
        dsc, solemc;
        measures=(accuracy, kappa)
    )
end
# 315.291 μs (3013 allocations: 214.52 KiB)
# con Ref migliora leggermente
# 314.580 μs (2992 allocations: 212.86 KiB)

@btime symbolic_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)
# 451.912 μs (3370 allocations: 247.12 KiB)

# Qualcosina è migliorato, però abbiamo visto che è sulle random forest che sole
# perde troppo rispetto a MLJ. vediamo ora come si comporta.

dsc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)
solemc = paso_test(dsc)
model_new = symbolic_analysis(
    dsc, solemc;
    measures=(accuracy, kappa)
)

# Esperimento originale
model_old = symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

@test model_new.measures.measures_values == model_old.measures.measures_values
# test superato

# Ora diamo un occhiata ai benchmark per vedere se effettivamente siamo migliorati
@btime begin
    setup_dataset(
        Xc, yc;
        model=RandomForestClassifier(),
        resample=Holdout(shuffle=true),
            train_ratio=0.7,
            rng=Xoshiro(1),   
    )
    solemc = paso_test(dsc)
    model_new = symbolic_analysis(
        dsc, solemc;
        measures=(accuracy, kappa)
    )
end
# 8.548 ms (96136 allocations: 4.92 MiB)
# e quadagnamo pure qui
# 7.896 ms (91382 allocations: 4.60 MiB)

@btime symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)
# 14.716 ms (183395 allocations: 9.30 MiB)

# funzionerà ancora? sulla carta potremmo dire di si, verifichiamolo con
# una battuta di test come fatto all'inizio:
function _pasorules(
    m::PasoDecisionEnsemble,
    # aggiunto arg i, root.info
    i::NamedTuple;
    suppress_parity_warning = true,
    kwargs...
)
    modelrules = [_pasorules(subm, i; kwargs...) for subm in models(m)]
    @assert all(r->consequent(r) isa ConstantModel, Iterators.flatten(modelrules))

    SoleModels.IterTools.imap(rulecombination->begin
        rulecombination = collect(rulecombination)
        ant = SoleModels.join_antecedents(antecedent.(rulecombination))
        # qui in bestguess ho dovuto usare 'nothing' al posto di m.weights
        # perchè ho iniziato ad usare, al posto della Union{Nothing, Vector},
        # Vector[] vettore vuoto al posto di Nothing.
        # Mi sembra più carino, anche perchè altrimenti il codice sarebbe pieno di Union{nothing, ...}
        # quindi dovrei modificare la bestgues per ceckare 'isempty' anzichè 'isnothing'
        # controllato il benchmark, sono identiche.
        # per ora basta passargli nothing, ma mi devo ricordare di rimettere m.weights! 
        o_cons = SoleModels.bestguess(outcome.(consequent.(rulecombination)), nothing; suppress_parity_warning)
        i_cons = merge(SoleModels.info.(consequent.(rulecombination))...)
        cons = ConstantModel(o_cons, i_cons)
        infos = merge(SoleModels.info.(rulecombination)...)
        Rule(ant, cons, infos)
        end, Iterators.product(modelrules...)
    )
end
_pasorules(m::PasoDecisionTree, i::NamedTuple; kwargs...) = _pasorules(pasoroot(m), i; kwargs...)

# Partiamo con DecisionTreeClassifier
for seed in 1:200
    dsc = setup_dataset(
        Xc, yc;
        model=DecisionTreeClassifier(),
        resample=Holdout(;shuffle=true),
        rng=Xoshiro(seed),
    )
    soleold = train_test(dsc)
    solenew = paso_test(dsc)

    modelold = soleold.sole[1]
    modelnew = solenew.sole[1]
    test_original = listrules(modelold)
    test_paso = pasorules(modelnew)

    @test length(test_original) == length(test_paso)
    for i in 1:length(test_paso)
        @test test_original[i].antecedent.grandchildren == test_paso[i].antecedent.grandchildren
        @test test_original[i].consequent.outcome == test_paso[i].consequent.outcome
    end
end

# proviamo anche con RandomForest
for seed in 1:50
    dsc = setup_dataset(
        Xc, yc;
        # con 100 alberi si rompe julia!
        model=RandomForestClassifier(n_trees=5),
        resample=Holdout(;shuffle=true),
        rng=Xoshiro(seed),
    )
    soleold = train_test(dsc)
    solenew = paso_test(dsc)

    modelold = soleold.sole[1]
    modelnew = solenew.sole[1]
    test_original = listrules(modelold)
    test_paso = pasorules(modelnew)

    @test length(test_original) == length(test_paso)
    for i in 1:length(test_paso)
        @test test_original[i].antecedent.grandchildren == test_paso[i].antecedent.grandchildren
        @test test_original[i].consequent.outcome == test_paso[i].consequent.outcome
    end
end
