# La prossima volta che mi si proporrà di referenziare una varabile,
# giuro espatrio.
# é stata una fatica enorme, ho dovuto rivedere molto codice,
# ma alla fine ce l'ho fatta.

# ---------------------------------------------------------------------------- #
# Prefazione:
# ---------------------------------------------------------------------------- #
# abbiamo visto che, nella costruzione dell'albero, o della foresta,
# ad ogni nodo e ad ogni foglia viene aggiunta una namedtuple 'info'
# contenente i vettori delle predizioni e delle label, in più,
# per eleganza di stampa, anche i vettori di feature names e class labels.
# Si tratta di una soluzione elegante ma pesantissima in termini di memoria
# e, soprattutto, estremamente ridondante in quanto la namedtuple 'info'
# è identica in ogni modo/foglia.

# ---------------------------------------------------------------------------- #
# Idee proposte:
# ---------------------------------------------------------------------------- #
# 1-eliminare la namedtuple 'info', tenere solo quella in 'root'.
# è sicuramente la soluzione più rapida, efficente ma rischiosa.
# Se abbiamo visto che funzioni come 'listrules' sono facilmente modificabili,
# per usare unicamente la 'info' in root,
# non posso dire, a priori, altrettanto per funzioni come 'readmetrics'
# le quali lavorano anche a livello di nodi e foglie, non solo di root.
# nel caso si decidesse di proseguire con questa soluzione,
# andrebbero riviste quelle funzioni.
# Il problema è che, come sempre a priori,
# io non posso sapere se qualcuno di voi utilizza le readmetrics magari
# per analizzare parte di un albero, magari senza per forza caricarlo tutto in memoria,
# e quindi non potendo avere accesso alla 'info' in root.

# 2-Referenziare 'info'.
# questa è la soluzione proposta da Gio e Marco che ho voluto seguire.
# sappiamo che è breaking, in quanto avremo un albero/foresta con un campo
# non più NamedTuple, ma Ref(NamedTuple), il che significa che tutto il codice 
# che lavora con il campo 'info' andrà rivisto.
# Però in questo caso la soluzione è più semplice del caso precedente perchè 
# non siamo più dipendenti dalla 'root': in ogni nodo/foglia
# avremo un riferimento alla 'info' di root,
# semplicemente dovremo accedervi con info[].campo_richiesto anzichè info.campo_richiesto.
# sembra facile.
# In più questa soluzione mantiene il concetto originale con cui avete sviluppato gli
# alberi in Sole e quindi, sulla carta, dovrebbe essere meno breaking.

# ---------------------------------------------------------------------------- #
# Referenziamo 'info'
# Difficoltà incontrate:
# ---------------------------------------------------------------------------- #
# 1-l'albero viene costruito in ricorsione.
# quindi ci troviamo di fronte ad un bel problema: se devo referenziare 'info' in 'root'
# è banale notare che la root sara istanziata per ultima, e quindi non è
# possibile referenziare qualcosa che vedrà la luce solo in futuro.

# 2-IPOTESI non VERIFICATA
# potrei creare, all'interno della funzione 'solemodels' una namedtuple 'info' vuota,
# referenziarla e passare la Ref a tutto l'albero.
# Non l'ho provata, ma concettualmente non mi piace molto:
# andrei a referenziare una variabile locale ad una funzione, che poi verrà cancellata
# una volta conclusa la funzione.
# Ne rimarrebbe il riferimento nell'albero, ma mi sembra tutt'altro che elegante.

# 3-Sentinella.
# parlando con Mauro, mi ha consigliato di usare un sentinella.
# Mi sembra una soluzione interessante ed elegante, che ho sviluppato in questo modo:
# - prima di creare l'albero, istanzio una struttura 'albero',
# quindi o una DecisionTree o una DecisionEnsemble, VUOTA.
# assegno a questa struttura il campo 'info' da referenziare.
# - costruisco l'albero in ricorsione, assegnando ad ogni nodo il riferimento alla 'info'
# già creata.
# - anzichè ritornare una struttura nuova, innesto l'albero nel campo a lui designato
# nella struttura precedentemente creata.

# 4-Ora quando eseguirò l'apply! sulla struttura, modificherà solamente 'info' di 'root',
# ma di reference tutte le modifiche vengono viste da ogni nodo/foglia.

# ---------------------------------------------------------------------------- #
# benchmark e considerazioni
# ---------------------------------------------------------------------------- #
# La modifica proposta funziona e abbiamo, come atteso, un crollo del consumo di memoria:
# in fondo al file troverete gli esperimenti di benchmark, qui commento solo i
# risultati:
# l'esperimento su Iris, RandomForest da 100 alberi, con il codice originale
# era di: 15.084 ms (183395 allocations: 9.30 MiB)
# mentre lo stesso esperimento, con gli stessi risultati, ma con le 'info' referenziate
# è di: 10.552 ms (104009 allocations: 5.84 MiB)
# un terzo più veloce, e quasi la metà delle allocazioni e memoria utilizzata.
# Niente male direi!
# Oltre a questo c'è un altro vantaggio enorme: 
# la forbice aumenta con l'aumentare della complessità del modello: 
# nel codice originale maggiore complessità = maggior numero di campi 'info'
# nel codice nuovo il campo 'info' è indipendente dalla complessità: è sempre e solo uno.

# Per vostra informazione: ho fatto anche la versione senza campi 'info': 
# ero curioso di vedere quanto meglio performasse.
# Effettivamente è un pochetto meglio, ma non abbastanza da proporvi questa soluzione.

# Fra tutte le prove che ho fatto, questa soluzione eccelle in termini di risparmio memoria.

# Se saremo tutti d'accordo, mi occuperò di fare una PR per SoleModels, non prima di averne anche verificato
# la solidità dei test.

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

const InfoStruct = (
    supporting_predictions=[],
    supporting_labels=[],
    featurenames=[],
    classlabels=[]
)

# ---------------------------------------------------------------------------- #
const Branch_or_Leaf{O} = Union{Branch{O}, LeafModel{O}}

default_parity_func = x->argmax(x)
DT_parity_func = x->first(sort(collect(keys(x))))

# to be used in case of UInt32
# default_parity_func = x->mode(x)
# DT_parity_func = x->mode(sort(x))

function set_aggregation(
    aggregation::Union{Nothing,Base.Callable}=nothing;
    parity_func::Base.Callable=default_parity_func
)::Base.Callable
    if isnothing(aggregation)
        aggregation = function (args...; suppress_parity_warning, kwargs...)
            SoleModels.bestguess(args...; suppress_parity_warning, parity_func, kwargs...)
        end
    end

    return aggregation
end

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
mutable struct PasoDecisionEnsemble{O} <: SoleModels.AbstractDecisionEnsemble{O}
    models::Union{Nothing,Vector{<:AbstractModel}}
    aggregation::Base.Callable
    weights::Union{Nothing,AbstractVector}
    info::Base.RefValue{<:NamedTuple}
    featim::Vector{<:Real}

    function PasoDecisionEnsemble{O}(
        models::AbstractVector{T},
        aggregation::Union{Nothing,Base.Callable},
        weights::Union{Nothing,AbstractVector},
        info::Base.RefValue{<:NamedTuple};
        suppress_parity_warning::Bool=false,
        parity_func::Base.Callable=x->argmax(x),
    ) where {O,T<:AbstractModel}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        if isnothing(aggregation)
            aggregation = function (args...; suppress_parity_warning, kwargs...) SoleModels.bestguess(args...; suppress_parity_warning, parity_func, kwargs...) end
        else
            !suppress_parity_warning || @warn "Unexpected value for suppress_parity_warning: $(suppress_parity_warning)."
        end
        # T = typeof(models)
        # W = typeof(weights)
        # A = typeof(aggregation)
        new{O}(collect(models), aggregation, weights, info, Float64[])
    end

    function PasoDecisionEnsemble{O}(
        info::Base.RefValue{<:NamedTuple},
        aggregation::Union{Nothing,Base.Callable}=nothing;
        featim::Vector{<:Real},
        parity_func::Base.Callable=default_parity_func
    ) where {O<:DT.Ensemble}
        new{O}(nothing, set_aggregation(aggregation; parity_func), nothing, info, featim)
    end
    
    function PasoDecisionEnsemble{O}(
        models::AbstractVector;
        kwargs...
    ) where {O}
        info = InfoStruct
        PasoDecisionEnsemble{O}(models, nothing, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        info::Base.RefValue{<:NamedTuple};
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        aggregation::Union{Nothing,Base.Callable},
        info::Base.RefValue{<:NamedTuple};
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, aggregation, nothing, info; kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        weights::AbstractVector,
        info::Base.RefValue{<:NamedTuple};
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
    root::Union{Nothing, R} where {R<:Union{LeafModel{O},PasoBranch{O}}}
    info::Base.RefValue{<:NamedTuple}

    function PasoDecisionTree(
        root::Union{LeafModel{O},PasoBranch{O}},
        info::Base.RefValue{<:NamedTuple},
    ) where {O}
        new{O}(root, info)
    end

    function PasoDecisionTree{O}(
        info::Base.RefValue{<:NamedTuple},
    ) where {O}
        new{O}(nothing, info)
    end

    function PasoDecisionTree(
        root::Any,
        info::Base.RefValue{<:NamedTuple},
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
        info::Base.RefValue{<:NamedTuple},
    )
        posconsequent isa PasoDecisionTree && (posconsequent = root(posconsequent))
        negconsequent isa PasoDecisionTree && (negconsequent = root(negconsequent))
        return PasoDecisionTree(Branch(antecedent, posconsequent, negconsequent, info))
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
    model        :: DT.Ensemble{T,O};
    featurenames :: Vector{Symbol},
    classlabels  :: AbstractVector{<:SoleModels.Label},
    weights      :: Union{Nothing, Vector{<:Number}}=nothing,
    aggregation  :: Union{Nothing,Base.Callable}=nothing,
    parity_func  :: Base.Callable=DT_parity_func,
    suppress_parity_warning::Bool=false
)::PasoDecisionEnsemble where {T,O}
    isnothing(aggregation) && begin
        aggregation = function (args...; suppress_parity_warning, kwargs...)
            SoleModels.bestguess(args...; suppress_parity_warning, parity_func, kwargs...)
        end
    end

    classtype = eltype(classlabels)
    info = (;
        supporting_labels = classtype[],
        supporting_predictions = classtype[],
        featurenames,
        classlabels
    )

    solestruct = PasoDecisionEnsemble{typeof(model)}(Ref(info), aggregation; featim=model.featim, parity_func);
    trees = map(t -> pasomodel(t, solestruct.info; featurenames, classlabels), model.trees);
    models = wrap.(trees)
    solestruct.models = collect(models)
    isnothing(weights) || (solestruct.weights = weights)

    return solestruct
end

function pasomodel(
    tree           :: DT.InfoNode{T,O};
    featurenames   :: Union{Nothing,Vector{Symbol}}=nothing,
    classlabels  :: AbstractVector{<:SoleModels.Label}
)::PasoDecisionTree where {T,O}
    classtype = eltype(classlabels)
    info = (;
        supporting_labels = classtype[],
        supporting_predictions = classtype[],
        featurenames,
        classlabels
    )

    solestruct = PasoDecisionTree{eltype(classlabels)}(Ref(info))
    
    root = pasomodel(tree.node, solestruct.info; featurenames, classlabels)

    # PasoDecisionTree(root, info)
    solestruct.root = root

    return solestruct
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
pasoroot(m::PasoDecisionTree) = m.root
models(m::PasoDecisionEnsemble) = m.models

function set_predictions(
    info  :: Base.RefValue{<:NamedTuple},
    preds :: AbstractVector{T},
    y     :: AbstractVector{S}
)::Base.RefValue{<:NamedTuple} where {T,S<:SoleModels.Label}
    info[] = merge(info[], (supporting_predictions=preds, supporting_labels=y))
    return info
end

aggregation(m::PasoDecisionEnsemble) = m.aggregation
weights(m::PasoDecisionEnsemble) = m.weights
# Returns the aggregation function, patched by weights if the model has them.
function weighted_aggregation(m::PasoDecisionEnsemble)
    if isnothing(weights(m))
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
function xplorer_apply(
    ds :: SoleXplorer.DecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    classlabels = sort(MLJ.report(ds.mach).classes_seen)
    solem        = pasomodel(MLJ.fitted_params(ds.mach).tree; featurenames, classlabels)
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
    solem        = pasomodel(MLJ.fitted_params(ds.mach).forest; featurenames, classlabels)
    logiset      = scalarlogiset(X, allow_propositional = true)
    pasoapply!(solem, logiset, y)
    return solem
end

function _paso_test(ds::SoleXplorer.EitherDataSet)::SoleXplorer.SoleModel
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

    return SoleXplorer.SoleModel(ds, solemodel)
end

function paso_test(args...; kwargs...)::SoleXplorer.SoleModel
    ds = SoleXplorer._setup_dataset(args...; kwargs...)
    _paso_test(ds)
end

paso_test(ds::SoleXplorer.AbstractDataSet)::SoleXplorer.SoleModel = _paso_test(ds)

# ---------------------------------------------------------------------------- #
function paso_analysis(
    X::AbstractDataFrame,
    y::AbstractVector,
    w::SoleXplorer.OptVector = nothing;
    extractor::Union{Nothing,SoleXplorer.RuleExtractor}=nothing,
    measures::Tuple{Vararg{SoleXplorer.FussyMeasure}}=(),
    kwargs...
)::SoleXplorer.ModelSet
    ds = SoleXplorer._setup_dataset(X, y, w; kwargs...)
    solem = _paso_test(ds)
    SoleXplorer._symbolic_analysis(ds, solem; extractor, measures)
end

# ---------------------------------------------------------------------------- #
model_new = paso_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);

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

model_new = paso_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);

# Esperimento originale
model_old = symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);

@test model_new.measures.measures_values == model_old.measures.measures_values
# test superato

# Ora diamo un occhiata ai benchmark per vedere se effettivamente siamo migliorati
@btime paso_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);
# 10.552 ms (104009 allocations: 5.84 MiB)

@btime symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);
# 15.084 ms (183395 allocations: 9.30 MiB)
