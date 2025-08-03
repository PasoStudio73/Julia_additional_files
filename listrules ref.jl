using Test
using SoleXplorer
using MLJ
using DataFrames, Random
const SX = SoleXplorer

Xc, yc = @load_iris
Xc = DataFrame(Xc)

modelc = symbolic_analysis(Xc, yc)

####################################################################

mutable struct PasoDecisionTree{O} <: AbstractModel{O}
    root::M where {M<:Union{LeafModel{O},Branch{O}}}
    info::NamedTuple

    function PasoDecisionTree(
        root::Union{LeafModel{O},Branch{O}},
        info::NamedTuple = (;),
    ) where {O}
        new{O}(root, info)
    end

    function PasoDecisionTree(
        root::Any,
        info::NamedTuple = (;),
    )
        root = wrap(root)
        M = typeof(root)
        O = outcometype(root)
        @assert M <: Union{LeafModel{O},Branch{O}} "" *
            "Cannot instantiate PasoDecisionTree{$(O)}(...) with root of " *
            "type $(typeof(root)). Note that the should be either a LeafModel or a " *
            "Branch. " *
            "$(M) <: $(Union{LeafModel,Branch{<:O}}) should hold."
        new{O}(root, info)
    end

    function PasoDecisionTree(
        antecedent::Formula,
        posconsequent::Any,
        negconsequent::Any,
        info::NamedTuple = (;),
    )
        posconsequent isa PasoDecisionTree && (posconsequent = root(posconsequent))
        negconsequent isa PasoDecisionTree && (negconsequent = root(negconsequent))
        return PasoDecisionTree(Branch(antecedent, posconsequent, negconsequent, info))
    end
end

# ---------------------------------------------------------------------------- #
#                          nuova funzione solemodel                            #
# ---------------------------------------------------------------------------- #
function get_featurenames(tree::Union{DT.Ensemble, DT.InfoNode})
    if !hasproperty(tree, :info)
        throw(ArgumentError("Please provide featurenames."))
    end
    return tree.info.featurenames
end
get_classlabels(tree::Union{DT.Ensemble, DT.InfoNode})::Vector{<:SoleModels.Label} = tree.info.classlabels

function get_condition(featid, featval, featurenames)
    test_operator = (<)
    feature = isnothing(featurenames) ? VariableValue(featid) : VariableValue(featid, featurenames[featid])
    return ScalarCondition(feature, test_operator, featval)
end

function pasomodel(
    tree           :: DT.InfoNode{T,O};
    featurenames   :: Vector{Symbol}=Symbol[],
    keep_condensed :: Bool=false,
)::PasoDecisionTree where {T,O}
    isempty(featurenames) && (featurenames = get_featurenames(tree))
    classlabels  = hasproperty(tree.info, :classlabels) ? get_classlabels(tree) : SoleModels.Label[]

    root, info = begin
        if keep_condensed
            root = pasomodel(tree.node, featurenames; classlabels)
            # anche qui: niente info
            # info = (;
            #     apply_preprocess=(y -> UInt32(findfirst(x -> x == y, classlabels))),
            #     apply_postprocess=(y -> classlabels[y]),
            # )
            info = (;)
            root, info
        else
            root = pasomodel(tree.node, featurenames; classlabels)
            info = (;)
            root, info
        end
    end

    # info = merge(info, (;
    #         featurenames=featurenames,
    #         supporting_predictions=root.info[:supporting_predictions],
    #         supporting_labels=root.info[:supporting_labels],
    #     )
    # )

    PasoDecisionTree(root, info)
end

function pasomodel(
    tree         :: DT.Node,
    featurenames :: Vector{Symbol};
    classlabels  :: AbstractVector{<:SoleModels.Label}=SoleModels.Label[],
)::Branch
    cond = get_condition(tree.featid, tree.featval, featurenames)
    antecedent = Atom(cond)
    lefttree  = pasomodel(tree.left, featurenames; classlabels )
    righttree = pasomodel(tree.right, featurenames; classlabels )

    # a costo di ripetermi...
    # info = (;
    #     supporting_predictions = [lefttree.info[:supporting_predictions]..., righttree.info[:supporting_predictions]...],
    #     supporting_labels = [lefttree.info[:supporting_labels]..., righttree.info[:supporting_labels]...],
    # )
    info = (;)

    return Branch(antecedent, lefttree, righttree, info)
end

function pasomodel(
    tree         :: DT.Leaf,
                 :: Vector{Symbol};
    classlabels  :: AbstractVector{<:SoleModels.Label}=SoleModels.Label[]
)::ConstantModel
    prediction, labels = isempty(classlabels) ? 
        (tree.majority, tree.values) : 
        (classlabels[tree.majority], classlabels[tree.values])

    # ci siamo capiti
    # info = (;
    #     supporting_predictions = fill(prediction, length(labels)),
    #     supporting_labels = labels,
    # )
    info = (;)

    SoleModels.ConstantModel(prediction, info)
end

# ---------------------------------------------------------------------------- #
#                           nuova funzione apply!                              #
# ---------------------------------------------------------------------------- #
# ERROR: type NamedTuple has no field supporting_labels
# bisogna modificare leggermente l'apply! esistente in modo che non vada a cercare
# field che abbiamo volutamente lasciato vuoti
# e che accetti, almeno per ora, PasoDecisionTree
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

function pasoapply!(
    m::PasoDecisionEnsemble,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    mode = :replace,
    leavesonly = false,
    suppress_parity_warning = false,
    kwargs...
)
    y = SoleModels.__apply_pre(m, d, y)

    preds = hcat([pasoapply!(subm, d, y; mode, leavesonly, kwargs...) for subm in models(m)]...)

    preds = SoleModels.__apply_post(m, preds)

    preds = [
        weighted_aggregation(m)(preds[i,:]; suppress_parity_warning, kwargs...)
        for i in 1:size(preds,1)
    ]

    preds = SoleModels.__apply_pre(m, d, preds)
    # return __pasoapply!(m, mode, preds, y, leavesonly)
    # giunto a questo punto io avrò tutte le predizioni finali,
    # quindi è giunto il momento di scrivere in root la struttura info
    # con supporting_labels e supporting_predictions.
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
    y = SoleModels.__apply_pre(m, d, y)
    preds = pasoapply!(pasoroot(m), d, y;
        mode = mode,
        leavesonly = leavesonly,
        kwargs...
    )
    # return __pasoapply!(m, mode, preds, y, leavesonly)
    # giunto a questo punto io avrò tutte le predizioni finali,
    # quindi è giunto il momento di scrivere in root la struttura info
    # con supporting_labels e supporting_predictions.
    m.info  = set_predictions(m.info, preds, y)
    return nothing
end

function pasoapply!(
    m::Branch,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    mode = :replace,
    leavesonly = false,
    # show_progress = true,
    kwargs...
)
    # @assert length(y) == ninstances(d) "$(length(y)) == $(ninstances(d))"
    if mode == :replace
        # non è più  necessario: si parte già con tutto vuoto
        # SoleModels.recursivelyemptysupports!(m, leavesonly)
        mode = :append
    end
    checkmask = SoleModels.checkantecedent(m, d, check_args...; check_kwargs...)
    preds = Vector{outputtype(m)}(undef,length(checkmask))
    @sync begin
        if any(checkmask)
            l = Threads.@spawn pasoapply!(
                posconsequent(m),
                slicedataset(d, checkmask; return_view = true),
                y[checkmask];
                check_args = check_args,
                check_kwargs = check_kwargs,
                mode = mode,
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
                mode = mode,
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
    return __pasoapply!(m, mode, preds, y, leavesonly)
end

function pasoapply!(
    m::ConstantModel,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    mode = :replace,
    leavesonly = false,
    kwargs...
)
    if mode == :replace
        # non serve più
        # SoleModels.recursivelyemptysupports!(m, leavesonly)
        mode = :append
    end
    preds = fill(outcome(m), ninstances(d))

    return __pasoapply!(m, mode, preds, y, leavesonly)
end

# questa funzione scrive la predizione nel campo 'supporting_prediction'
# che attualmente non è presente.
# quello che vorrei fare io, nell'apply, non è riempire tutte le 'info', ma propagare la 
# predizione fino alla root
function __pasoapply!(m, mode, preds, y, leavesonly)
    if !leavesonly || m isa LeafModel
        if mode == :replace
            if haskey(m.info, :supporting_predictions)
                # empty!(m.info.supporting_predictions)
                # append!(m.info.supporting_predictions, preds)
            end
            # empty!(m.info.supporting_labels)
            # append!(m.info.supporting_labels, y)
        elseif mode == :append
            if haskey(m.info, :supporting_predictions)
                # append!(m.info.supporting_predictions, preds)
            end
            # append!(m.info.supporting_labels, y)
        else
            error("Unexpected apply mode: $mode.")
        end
    end
    preds = SoleModels.__apply_post(m, preds)

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

@btime symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)
# 14.716 ms (183395 allocations: 9.30 MiB)

# Come potete vedere, se usando alberi decisionali i miglioramenti di prestazioni sono pressochè
# trascurabili, se passiamo a strutture più complesse, come le foreste,
# i miglioramenti sono enormi.
# E questo su un esperimento giocattolo, pensatelo su un esperimento reale!
# Non è finita qui però: rimane ancora il problema della listrule:
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

# ---------------------------------------------------------------------------- #
#                                  Conclusione                                 #
# ---------------------------------------------------------------------------- #
# abbiamo dimostrato che SoleModels può essere migliorato in termini di efficenza
# con ampio margine, senza che le funzionalità da me trovate, vengano compromesse.

# C'è però l'ultimo punto: ed è il vostro benestare.

# Ho totalmente stravolto la creazione di un albero sole andando ad intaccare quella 
# che è la sua struttura base.
# Ne ho scalfito la superficie sperando che non vi fosse nulla sotto.
# Ora chiedo a voi se questo lavoro ha senso, è pericoloso, oppure va ad intaccare funzionalità
# che non ho preso in considerazione per mera ignoranza.

# Vi ringrazio anticipatamente per la pazienza e per qualsiasi dubbio o domanda
# non esitate a scrivermi.
# Se siete in vacanza guai a voi se mi scrivete, riposatevi!


# qui ho lasciato del codice commentato che forse potrà servire in futuro
# ---------------------------------------------------------------------------- #
#                     DecisionTree apply from DataFrame X                      #
# ---------------------------------------------------------------------------- #
# get_featid(s::Branch) = s.antecedent.value.metacond.feature.i_variable
# get_cond(s::Branch)   = s.antecedent.value.metacond.test_operator
# get_thr(s::Branch)    = s.antecedent.value.threshold

# function set_predictions(
#     info  :: NamedTuple,
#     preds :: Vector{T},
#     y     :: AbstractVector{S}
# )::NamedTuple where {T,S<:SoleModels.Label}
#     merge(info, (supporting_predictions=preds, supporting_labels=y))
# end

# function pasoapply!(
#     solem :: PasoDecisionEnsemble{O,T,A,W},
#     X     :: AbstractDataFrame,
#     y     :: AbstractVector;
#     suppress_parity_warning::Bool=false
# )::Nothing where {O,T,A,W}
#     predictions = permutedims(hcat([pasoapply(s, X, y) for s in get_models(solem)]...))
#     predictions = aggregate(solem, predictions, suppress_parity_warning)
#     solem.info  = set_predictions(solem.info, predictions, y)
#     return nothing
# end

# function pasoapply!(
#     solem :: PasoDecisionTree{T},
#     X     :: AbstractDataFrame,
#     y     :: AbstractVector{S}
# )::Nothing where {T, S<:SoleModels.Label}
#     predictions = [pasoapply(solem.root, x) for x in eachrow(X)]
#     solem.info  = set_predictions(solem.info, predictions, y)
#     return nothing
# end

# function pasoapply(
#     solebranch :: Branch{T},
#     X          :: AbstractDataFrame,
#     y          :: AbstractVector{S}
# ) where {T, S<:SoleModels.Label}
#     predictions     = SoleModels.Label[pasoapply(solebranch, x) for x in eachrow(X)]
#     solebranch.info = set_predictions(solebranch.info, predictions, y)
#     return predictions
# end

# function pasoapply(
#     solebranch :: Branch{T},
#     x          :: DataFrameRow
# )::T where T
#     featid, cond, thr = get_featid(solebranch), get_cond(solebranch), get_thr(solebranch)
#     feature_value     = x[featid]
#     condition_result  = cond(feature_value, thr)
    
#     return condition_result ?
#         pasoapply(solebranch.posconsequent, x) :
#         pasoapply(solebranch.negconsequent, x)
# end

# function pasoapply(leaf::ConstantModel{T}, ::DataFrameRow)::T where T
#     leaf.outcome
# end
