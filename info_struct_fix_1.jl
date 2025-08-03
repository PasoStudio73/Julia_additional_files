# ---------------------------------------------------------------------------- #
#                                 prefazione                                   #
# ---------------------------------------------------------------------------- #
# Gio e Marco hanno proposto diverse migliorie/soluzioni a quanto proposto in precedenza.
# Per non rischiare di incartarmi, ho pensato di introdurle una ad una,
# presentandovi una serie di file/documentazione in ordine crescente di implementazione.

# ---------------------------------------------------------------------------- #
#                                Gio's fix #1                                  #
# ---------------------------------------------------------------------------- #
# Osservazione: l’informazione pesante nelle info può essere iper compressa. Ecco alcune idee

# Per quanto riguarda “features”, dovrebbe essere lo stesso vettore X tutti. Se lo è,
# puo essere allocato una sola volta e condiviso tra nodi diversi (forse è gia cosi?)
# Se non lo è, normalizzare in modo che ci sia un solo vettore Unione di tutti,
# e poi condividerlo; ma poi come viene usato questo vettore? È solo x visualizzazione forse?
# O ha un effetto sul behavior? Nel caso si potrebbe mettere il nome della feature
# direttamente nel i_var anziché l’indice.

# ---------------------------------------------------------------------------- #
#                               considerazioni                                 #
# ---------------------------------------------------------------------------- #
# Mi viene da ricordare che le featurenames fossero state inserite solo per questioni
# di stampa, e credo d'essere stato proprio io insieme a Gio ad aggiungerle alla struttura 'info',
# poco prima che partissimo per Pisa.
# La motivazione era di rendere subito più chiara la rappresentazione dell'albero:
# sostituire le varie V1, V2 ... Vn, con i nomi delle feature, per rendere da subito più
# intuitivo quali fossero le features che il modello prendeva in considerazione.

# A mio parere, le considerazioni che facemmo, rimangono tutt'oggi valide,
# quindi l'idea di salvare le 'featurenames' una sola volta, e non in ogni 'info',
# è giusta.

# ---------------------------------------------------------------------------- #
#                          PROPOSTA NUOVA STRUTTURA                            #
# ---------------------------------------------------------------------------- #
# Quello che sto per proporre è pesantemente BREAKING e, se accettato, mi costringerà
# a riscrivere molto codice.
# Le strutture che propongo di modificare sono, per ora:

# struct DecisionTree{O} <: AbstractModel{O}
#     root::M where {M<:Union{LeafModel{O},Branch{O}}}
#     info::NamedTuple

# struct DecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable,W<:Union{Nothing,AbstractVector}} <: AbstractDecisionEnsemble{O}
#     models::Vector{T}
#     aggregation::A
#     weights::W
#     info::NamedTuple

# e di conseguenza anche:

# struct DecisionXGBoost{O,T<:AbstractModel,A<:Base.Callable} <: AbstractDecisionEnsemble{O}
#     models::Vector{T}
#     aggregation::A
#     info::NamedTuple

# Marco ha proposto di rendere 'info' una Ref, ottima idea.
# A questo punto a me potrebbe venire in mente un wrapper per DecisionTree fatto più o meno così:

# struct SoleDecisionSet
#     models::Union{DecisionTree, DecisionEnsemble, DecisioneXGBoost}
#     info::Info
# end

# e modificare le precedenti struct in questo modo:

# struct DecisionTree{O} <: AbstractModel{O}
#     root::M where {M<:Union{LeafModel{O},Branch{O}}}

# struct DecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable,W<:Union{Nothing,AbstractVector}} <: AbstractDecisionEnsemble{O}
#     models::Vector{T}
#     aggregation::A
#     weights::W

# struct DecisionXGBoost{O,T<:AbstractModel,A<:Base.Callable} <: AbstractDecisionEnsemble{O}
#     models::Vector{T}
#     aggregation::A

# Mi scuso: la struttura DecisionXGBoost è assolutamente ridondante ed è stata creata solo per i 
# successivi apply specifici.
# Allora non avevo ancora dimestichezza con la tipizzazione di julia, ora mi rendo
# conto che può essere eliminata e posso modificare gli apply specifici per in tipo.
# Da fare!

# ---------------------------------------------------------------------------- #
#                          preparazione esperimento                            #
# ---------------------------------------------------------------------------- #
using SoleXplorer
using SoleModels
using MLJ
using DataFrames, Random
using Test, BenchmarkTools
import DecisionTree as DT

# Per curiosità voglio verificare la struttura di ogni modello importato dal pacchetto "DecisionTree",
# In classificazione uso il classico dataset Iris, mentre in regressione ho scelto
# il dataset Boston, entrambi presenti in MLJ.

Xc, yc = @load_iris
Xc = DataFrame(Xc)

featurenames = unique(yc)

model_dtc = symbolic_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

model_rfc = symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

model_abc = symbolic_analysis(
    Xc, yc,
    model=AdaBoostStumpClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

model_dtr = symbolic_analysis(
    Xr, yr,
    model=DecisionTreeRegressor(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(rms, l2)
)

model_rfr = symbolic_analysis(
    Xr, yr,
    model=RandomForestRegressor(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(rms, l2)
)

sole_dtc = solemodels(model_dtc)
sole_rfc = solemodels(model_rfc)
sole_abc = solemodels(model_abc)
sole_dtr = solemodels(model_dtr)
sole_rfr = solemodels(model_rfr)

# ---------------------------------------------------------------------------- #
#              NUOVE STRUTTURE DecisionEnsemble e DecisionTree                 #
# ---------------------------------------------------------------------------- #
# portando fuori la struttura 'info' dalla struttura albero,
# non è più necessario averla mutable, può tornare ad essere immutabile.
# Ho creato delle costanti, non per questioni di efficenza,
# ma solo per mera eleganza

const Optional{T}  = Union{T, Nothing}
const OptCallable  = Optional{Base.Callable}
# const OptAbsVector = Optional{AbstractVector}
const OptAbsVector{T} = Optional{AbstractVector{T}}

const Branch_or_Leaf{O} = Union{Branch{O}, LeafModel{O}}

# const default_parity_func = x->argmax(x)
# const DT_parity_func = x->first(sort(collect(keys(x))))

default_parity_func = x->argmax(x)
DT_parity_func = x->first(sort(collect(keys(x))))

struct PasoDecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable,W<:Real} <: SoleModels.AbstractDecisionEnsemble{O}
    models::Vector{T}
    aggregation::A
    weights::Vector{W}

    function PasoDecisionEnsemble{O}(
        models::AbstractVector{T};
        aggregation::OptCallable=nothing,
        weights::OptAbsVector{W}=nothing,
        suppress_parity_warning::Bool=false,
        parity_func::Base.Callable=default_parity_func
    ) where {O,T<:AbstractModel, W<:Real}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)

        if isnothing(aggregation)
            aggregation = function(args...; suppress_parity_warning, kwargs...)
                SoleModels.bestguess(args...; suppress_parity_warning, parity_func, kwargs...)
            end
        else
            !suppress_parity_warning || @warn "Unexpected value for suppress_parity_warning: $(suppress_parity_warning)."
        end

        A = typeof(aggregation)
        new{O,T,A,W}(collect(models), aggregation, weights)
    end

    function PasoDecisionEnsemble(models::AbstractVector{Any}; kwargs...)
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        O = Union{outcometype.(models)...}
        PasoDecisionEnsemble{O}(models; kwargs...)
    end
end

struct PasoDecisionTree{O} <: AbstractModel{O}
    root::M where {M<:Branch_or_Leaf{O}}

    function PasoDecisionTree(root::Branch_or_Leaf{O}) where {O}
        new{O}(root)
    end

    function PasoDecisionTree(root::Any)
        root = wrap(root)
        M = typeof(root)
        O = outcometype(root)
        @assert M <: Union{LeafModel{O},Branch{O}} "" *
            "Cannot instantiate PasoDecisionTree{$(O)}(...) with root of " *
            "type $(typeof(root)). Note that the should be either a LeafModel or a " *
            "Branch. " *
            "$(M) <: $(Union{LeafModel,Branch{<:O}}) should hold."
        new{O}(root)
    end

    function PasoDecisionTree(
        antecedent::Formula,
        posconsequent::Any,
        negconsequent::Any,
    )
        posconsequent isa PasoDecisionTree && (posconsequent = root(posconsequent))
        negconsequent isa PasoDecisionTree && (negconsequent = root(negconsequent))
        return PasoDecisionTree(Branch(antecedent, posconsequent, negconsequent))
    end
end

# ---------------------------------------------------------------------------- #
#                                Gio's fix #2                                  #
# ---------------------------------------------------------------------------- #
# Numero due, più importante: supporting labels e predictions sono veramente inutilmente heavy.
# Alcune idee x comprimerle.
# Primo step per migliorare è usare categorical vectors anziché vettori di stringhe come sono ora
# (questo migliora solo il caso classificazione).

# Partiamo dal caso di classificazione e vediamo qualche benchmark:
# Metodo usato in SoleXplorer:
@btime begin
    label_levels = categorical(sole_dtc[1].info.supporting_labels, levels=levels(sole_dtc[1].info.supporting_labels))
    label_coded = @. MLJ.levelcode(label_levels)
end
# 13.858 μs (82 allocations: 4.31 KiB)

@btime begin
    label_levels = categorical(sole_dtc[1].info.supporting_labels, levels=sort(unique(sole_dtc[1].info.supporting_labels)))
    label_coded = @. MLJ.levelcode(label_levels)
end
# 2.420 μs (23 allocations: 1.77 KiB)

@btime begin
    label_levels = categorical(sole_dtc[1].info.supporting_labels, levels=sort(unique(sole_dtc[1].info.supporting_labels)))
    label_coded = label_levels.refs
end
# 2.355 μs (20 allocations: 1.33 KiB)

label_levels = categorical(sole_dtc[1].info.supporting_labels, levels=levels(sole_dtc[1].info.supporting_labels))
label_coded_1 = @. MLJ.levelcode(label_levels)

label_levels = categorical(sole_dtc[1].info.supporting_labels, levels=sort(unique(sole_dtc[1].info.supporting_labels)))
label_coded_2 = @. MLJ.levelcode(label_levels)

label_levels = categorical(sole_dtc[1].info.supporting_labels, levels=sort(unique(sole_dtc[1].info.supporting_labels)))
label_coded_3 = label_levels.refs

@test label_coded_1 == label_coded_2
@test label_coded_1 == Int64.(label_coded_3)

@btime sort(Symbol.(sole_dtc[1].info.supporting_labels))
# 3.392 μs (7 allocations: 1.30 KiB)

@btime begin
    clabels = Symbol.(sole_dtc[1].info.supporting_labels)
    sort!(clabels)
end
# 3.206 μs (5 allocations: 896 bytes)

# ---------------------------------------------------------------------------- #
#                            NUOVA STRUTTURA Info                              #
# ---------------------------------------------------------------------------- #
# Questa struttura andrebbe a sostituire la vecchia NamedTuple Info,
# e include la seconda idea di Gio riguardo alla classificazione,
# cioè salvare labels e predictions come refs di categorical.
# Avendo cura di salvarsi le classlabels ordinate, altrimenti non si può tornare indietro.
# Le classlabels, così come le già viste featurenames,
# In questo punto potrebbero servire solo per la print (per ora uso il condizionale)

abstract type AbstractInfo{T,O} end

function get_refs(y::AbstractVector{<:SoleModels.CLabel})::Vector{UInt32}
    if y isa MLJ.CategoricalArray
        return y.refs
    else
        cat_array = categorical(y, levels=sort(unique(y)))
        return cat_array.refs
    end
end

struct Info{T,O} <: AbstractInfo{T,O}
    labels       :: Vector{T}
    predictions  :: Vector{T}
    featurenames :: OptAbsVector{Symbol}
    classlabels  :: OptAbsVector{Symbol}

    # Generic constructor
    function Info{O}(
        labels       :: Vector{T},
        predictions  :: Vector{T};
        featurenames :: OptAbsVector,
        classlabels  :: OptAbsVector
    )::Info where {T,O<:SoleModels.Label}
        fnames = if isnothing(featurenames)
            nothing
        else
            eltype(featurenames) <: Symbol ? featurenames : Symbol.(featurenames)
        end

        clabels = if isnothing(classlabels)
            nothing
        else
            eltype(classlabels)  <: Symbol ? classlabels  : Symbol.(classlabels)
        end
        sort!(clabels)

        new{T, O}(labels, predictions, fnames, clabels)
    end

    # Classification constructor
    function Info(
        labels      :: AbstractVector{L},
        predictions :: AbstractVector{L};
        kwargs...
    )::Info where {L<:SoleModels.CLabel}
        labels_refs = get_refs(labels)
        preds_refs  = get_refs(predictions)
        Info{eltype(labels)}(labels_refs, preds_refs; kwargs...)
    end

    # Regression constructor
    function Info(
        labels       :: AbstractVector{L},
        predictions  :: AbstractVector{L};
        float32      :: Bool=False;
        kwargs...
    )::Info where {L<:SoleModels.RLabel}
        float32 ?
            Info{eltype(labels)}(Float32.(labels), Float32.(predictions); kwargs...) :
            Info{eltype(labels)}(labels, predictions; kwargs...)
    end
end

# prova
labels = sole_dtc[1].info.supporting_labels
predictions = sole_dtc[1].info.supporting_predictions
feturenames = MLJ.report(model_dtc.ds.mach).features
classlabels = MLJ.report(model_dtc.ds.mach).classes_seen

test = Info(labels, predictions; featurenames, classlabels)

# La cosa che mi piace è che concettualmente abbiamo diviso la creazione dell'albero,
# dalla sua applicazione, anche a livello di strutture dati:
# ora 'solemodel' si occupa esclusivamente di creare la struttura DecisionTree/Ensemble,
# mentre 'apply' usa la struttura Decision, ma crea la struttura Info
# e magari le combina insieme in una sopra struttura?

struct SoleModel
    model::Union{PasoDecisionEnsemble, PasoDecisionTree}
    info::Info
end

# ---------------------------------------------------------------------------- #
#                          NUOVA FUNZIONE solemodel                            #
# ---------------------------------------------------------------------------- #
# Come si potrà notare, ora 'solemodel' è ultra semplificata: si occupa esclusivamente
# di convertire l'albero o la forest DecisionTree in un modello Sole.
# La semplificazione più evidente è il totale inutilizzo di featurenames e classnames.
# Per 2 motivi:
# 'info' non c'è più; se ne occuperà la funzione 'apply!' di creare la struttura 'info'.
# la 'print' di questa struttura restituirà un albero con features senza nome (V1, V2 .. Vn)
# e le classlabels saranno quelle originali, quindi nel caso di DecisionTree, numeri interi.
# In teoria questo sarebbe in contrasto con quanto detto qualche riga fa,
# ma ho pensato che una print fatta come si deve la si potrà fare una volta che avremo pronta anche la struttura info,
# quindi dopo l'apply.
# E che quindi sarà conveniente fare una struttura che contenga sia solemodel che info,
# e la stampa di quest'ultima, sarà completa.

function get_condition(featid, featval)
    test_operator = (<)
    feature = VariableValue(featid)
    ScalarCondition(feature, test_operator, featval)
end

function pasomodel(
    model::DT.Ensemble{T,O};
    weights::AbstractVector=Vector{Float64}(),
    parity_func::Base.Callable=DT_parity_func
)::PasoDecisionEnsemble where {T,O}
    trees = map(t -> pasomodel(t), model.trees)

    isempty(weights) ?
        PasoDecisionEnsemble{O}(trees; parity_func) :
        PasoDecisionEnsemble{O}(trees, weights; parity_func)
end

function pasomodel(tree::DT.InfoNode{T,O};)::PasoDecisionTree where {T,O}
    root = pasomodel(tree.node)
    PasoDecisionTree(root)
end

function pasomodel(tree::DT.Node)::Branch
    cond = get_condition(tree.featid, tree.featval)
    antecedent = Atom(cond)
    lefttree  = pasomodel(tree.left)
    righttree = pasomodel(tree.right)
    Branch(antecedent, lefttree, righttree)
end

function pasomodel(tree::DT.Leaf)::ConstantModel
    SoleModels.ConstantModel(tree.majority)
end

# ---------------------------------------------------------------------------- #
#                            NUOVA funzione apply!                             #
# ---------------------------------------------------------------------------- #
# una differenza fondamentale rispetto alla precedente versione è che gia dall'apply! dobbiamo
# differenziare tra classificazione e regressione, in quanto gli info sono differenti (classificazione/regressione)
# la differenza principale sta nelle classlabels: ovviamente inutili in regressione.
# ma in regressione possiamo scegliere se salvare i dati delle labels in float32 (disabilitato di default, almeno per ora)

# non potrà essere più apply!, ma solo apply, questo perchè il suo output sara la struttura Info

pasoroot(m::PasoDecisionTree) = m.root
models(m::PasoDecisionEnsemble) = m.models

# function set_predictions(
#     info  :: NamedTuple,
#     preds :: AbstractVector{T},
#     y     :: AbstractVector{S}
# )::NamedTuple where {T,S<:SoleModels.Label}
#     merge(info, (supporting_predictions=preds, supporting_labels=y))
# end

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

# QUESTA TIPIZZAZIONE FA CAGARE:
# devo salvare il tipo di provenienza in m, esempio DecisionEnsemble{XGBoostDecisinTreeClassifier}
# altrimenti col cavolo che riesco a fare gli apply come vorrei
function pasoapply(
    m::PasoDecisionEnsemble,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector{<:SoleModels.CLabel};
    featurenames::AbstractVector,
    classlabels::AbstractVector,
    # mode = :replace,
    leavesonly = false,
    suppress_parity_warning = false,
    kwargs...
)::Info
    # mi sento di eliminare __apply_pre e __apply_post
    # in quanto, da un primo sguardo, sembrano avere senso sono nel caso in cui le classlabels
    # non siano presenti.
    # ma da come stiamo rivedendo la funzione, le classlabel devono essere sempre presenti.
    # questo ovviamente solo nel caso di classificazione.
    # y = SoleModels.__apply_pre(m, d, y)

    preds = hcat([pasoapply(subm, d, y; mode, leavesonly, kwargs...) for subm in models(m)]...)

    # preds = SoleModels.__apply_post(m, preds)

    preds = map(eachrow(preds)) do row
        weighted_aggregation(m)(row; suppress_parity_warning, kwargs...)
    end

    # preds = SoleModels.__apply_pre(m, d, preds)

    # m.info  = set_predictions(m.info, preds, y)
    # return nothing
    Info(y, preds; featurenames, classlabels)
end

function pasoapply(
    m::PasoDecisionTree,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector{<:SoleModels.CLabel};
    featurenames::AbstractVector,
    classlabels::AbstractVector,
    # mode = :replace,
    leavesonly = false,
    kwargs...
)
    # y = SoleModels.__apply_pre(m, d, y)
    preds = pasoapply(pasoroot(m), d, y;
        # mode = mode,
        leavesonly,
        kwargs...
    )

    # m.info  = set_predictions(m.info, preds, y)
    # return nothing
    Info(y, preds; featurenames, classlabels)
end

function pasoapply(
    m::Branch,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    # mode = :replace,
    leavesonly = false,
    # show_progress = true,
    kwargs...
)
    # if mode == :replace
    #     mode = :append
    # end
    checkmask = SoleModels.checkantecedent(m, d, check_args...; check_kwargs...)
    preds = Vector{outputtype(m)}(undef,length(checkmask))
    @sync begin
        if any(checkmask)
            l = Threads.@spawn pasoapply(
                posconsequent(m),
                slicedataset(d, checkmask; return_view = true),
                y[checkmask];
                check_args,
                check_kwargs,
                # mode = mode,
                leavesonly,
                kwargs...
            )
        end
        ncheckmask = (!).(checkmask)
        if any(ncheckmask)
            r = Threads.@spawn pasoapply(
                negconsequent(m),
                slicedataset(d, ncheckmask; return_view = true),
                y[ncheckmask];
                check_args,
                check_kwargs,
                # mode = mode,
                leavesonly,
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
    # return __pasoapply(m, mode, preds, y, leavesonly)
    return preds
end

function pasoapply(
    m::ConstantModel,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    # mode = :replace,
    leavesonly = false,
    kwargs...
)
    # if mode == :replace
    #     # non serve più
    #     # SoleModels.recursivelyemptysupports!(m, leavesonly)
    #     mode = :append
    # end
    return fill(outcome(m), ninstances(d))

    # return __pasoapply(m, mode, preds, y, leavesonly)
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
    classlabels = MLJ.report(ds.mach).classes_seen
    solem        = pasomodel(MLJ.fitted_params(ds.mach).tree)
    logiset      = scalarlogiset(X, allow_propositional = true)
    @show typeof(y)
    info = pasoapply(solem, logiset, y; featurenames, classlabels)
    return solem, info
end

function xplorer_apply(
    ds :: SoleXplorer.PropositionalDataSet{RandomForestClassifier},
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    classlabels  = ds.mach.fitresult[2]
    solem        = pasomodel(MLJ.fitted_params(ds.mach).forest)
    logiset      = scalarlogiset(X, allow_propositional = true)
    info = pasoapply(solem, logiset, y; featurenames, classlabels)
    return solem, info
end

function _paso_test(ds::SoleXplorer.EitherDataSet)::Vector{SoleModel}
    n_folds   = length(ds.pidxs)
    solemodel = Vector{SoleModel}(undef, n_folds)

    # TODO this can be parallelizable
    @inbounds for i in 1:n_folds
        train, test = get_train(ds.pidxs[i]), get_test(ds.pidxs[i])
        X_test = @views get_X(ds)[test, :]
        y_test = get_y(ds)[test]

        SoleXplorer.has_xgboost_model(ds) && SoleXplorer.set_watchlist!(ds, i)

        MLJ.fit!(ds.mach, rows=train, verbosity=0)
        solem, info = xplorer_apply(ds, X_test, y_test)
        solemodel[i] = SoleModel(solem, info)
    end

    return solemodel
end

function paso_test(args...; kwargs...)::Vector{SoleModel}
    ds = SoleXplorer._setup_dataset(args...; kwargs...)
    _paso_test(ds)
end

paso_test(ds::SoleXplorer.AbstractDataSet)::Vector{SoleModel} = _paso_test(ds)

# Verifichiamo il corretto funzionamento
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)
solemc = paso_test(dsc)

rfc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)
solemc = paso_test(rfc)

# ---------------------------------------------------------------------------- #
#                               get operations                                 #
# ---------------------------------------------------------------------------- #
function get_operations(
    measures   :: Vector,
    prediction :: Symbol,
)
    map(measures) do m
        kind_of_proxy = MLJ.MLJBase.StatisticalMeasuresBase.kind_of_proxy(m)
        observation_scitype = MLJ.MLJBase.StatisticalMeasuresBase.observation_scitype(m)
        isnothing(kind_of_proxy) && (return paso_predict)

        if prediction === :probabilistic
            if kind_of_proxy === MLJ.MLJBase.LearnAPI.Distribution()
                return paso_predict
            elseif kind_of_proxy === MLJ.MLJBase.LearnAPI.Point()
                if observation_scitype <: Union{Missing,Finite}
                    return paso_predict_mode
                elseif observation_scitype <:Union{Missing,Infinite}
                    return paso_predict_mean
                else
                    throw(err_ambiguous_operation(prediction, m))
                end
            else
                throw(err_ambiguous_operation(prediction, m))
            end
        elseif prediction === :deterministic
            if kind_of_proxy === MLJ.MLJBase.LearnAPI.Distribution()
                throw(err_incompatible_prediction_types(prediction, m))
            elseif kind_of_proxy === MLJ.MLJBase.LearnAPI.Point()
                return paso_predict
            else
                throw(err_ambiguous_operation(prediction, m))
            end
        elseif prediction === :interval
            if kind_of_proxy === MLJ.MLJBase.LearnAPI.ConfidenceInterval()
                return paso_predict
            else
                throw(err_ambiguous_operation(prediction, m))
            end
        else
            throw(MLJ.MLJBase.ERR_UNSUPPORTED_PREDICTION_TYPE)
        end
    end
end

# ---------------------------------------------------------------------------- #
#                                eval measures                                 #
# ---------------------------------------------------------------------------- #
get_preds(m::SoleModel) = m.info.predictions
get_labels(m::SoleModel) = m.info.labels

function paso_predict(preds::Vector{UInt32}, ::Vector{UInt32})
    # eltype(preds) <: CLabel ?
    #     begin
    #         classes_seen = unique(labels)
    #         eltype(preds) <: MLJ.CategoricalValue ||
    #             (preds = categorical(preds, levels=levels(classes_seen)))
            [UnivariateFinite([p], [1.0]) for p in preds]
        # end :
        # preds
end

paso_predict_mode(preds::Vector{UInt32}, ::Vector{UInt32}) = preds

function eval_measures(
    ds::SoleXplorer.EitherDataSet,
    solem::Vector{SoleModel},
    measures::Tuple{Vararg{SoleXplorer.FussyMeasure}},
    y_test::Vector{<:AbstractVector{<:SoleModels.Label}}
)::SoleXplorer.Measures
    mach_model = SoleXplorer.get_mach_model(ds)
    measures        = MLJ.MLJBase._actual_measures([measures...], mach_model)
    operations      = get_operations(measures, MLJ.MLJBase.prediction_type(mach_model))

    nfolds          = length(ds)
    test_fold_sizes = [length(y_test[k]) for k in 1:nfolds]
    nmeasures       = length(measures)

    # weights used to aggregate per-fold measurements, which depends on a measures
    # external mode of aggregation:
    fold_weights(mode) = nfolds .* test_fold_sizes ./ sum(test_fold_sizes)
    fold_weights(::MLJ.MLJBase.StatisticalMeasuresBase.Sum) = nothing
    
    measurements_vector = mapreduce(vcat, 1:nfolds) do k
        # Get the classlabels from the Info struct
        classlabels = solem[k].info.classlabels
        
        yhat_given_operation = Dict(op => begin
            preds_refs = op(get_preds(solem[k]), get_labels(solem[k]))
            # Convert UInt32 refs back to string labels using classlabels
            if eltype(preds_refs) <: UInt32 && !isempty(classlabels)
                String.(classlabels[preds_refs])
            else
                preds_refs
            end
        end for op in unique(operations))
        # yhat_given_operation = Dict(op=>op(get_preds(solem[k]), get_labels(solem[k])) for op in unique(operations))

        # costretto a convertirlo a stringa in quanto certe misure di statistical measures non accettano
        # categorical array, tipo confusion matrix e kappa
        test = eltype(y_test[k]) <: SoleModels.CLabel ? String.(y_test[k]) : y_test[k]

        [map(measures, operations) do m, op
            m(
                yhat_given_operation[op],
                test,
                # MLJ.MLJBase._view(weights, test),
                # class_weights
                MLJ.MLJBase._view(nothing, test),
                nothing
            )
        end]
    end

    measurements_matrix = permutedims(reduce(hcat, measurements_vector))

    # measurements for each fold:
    fold = map(1:nmeasures) do k
        measurements_matrix[:,k]
    end

    # overall aggregates:
    measures_values = map(1:nmeasures) do k
        m = measures[k]
        mode = MLJ.MLJBase.StatisticalMeasuresBase.external_aggregation_mode(m)
        MLJ.MLJBase.StatisticalMeasuresBase.aggregate(
            fold[k];
            mode,
            weights=fold_weights(mode)
        )
    end

    SoleXplorer.Measures(fold, measures, measures_values, operations)
end

# ---------------------------------------------------------------------------- #
#                              symbolic_analysis                               #
# ---------------------------------------------------------------------------- #
mutable struct ModelSet <: SoleXplorer.AbstractModelSet
    ds       :: SoleXplorer.EitherDataSet
    sole     :: Vector{SoleModel}
    rules    :: SoleXplorer.OptRules
    measures :: SoleXplorer.OptMeasures
end

function _paso_analysis(
    ds::SoleXplorer.EitherDataSet,
    solem::Vector{SoleModel};
    extractor::Union{Nothing,SoleXplorer.RuleExtractor}=nothing,
    measures::Tuple{Vararg{SoleXplorer.FussyMeasure}}=(),
)::ModelSet
    # rules = isnothing(extractor)  ? nothing : begin
    #     # TODO propaga rng, dovrai fare intrees mutable struct
    #     extractrules(extractor, ds, solem)
    # end

    measures = isempty(measures) ? nothing : begin
        y_test = SoleXplorer.get_y_test(ds)
        # all_classes = unique(Iterators.flatten(y_test))
        eval_measures(ds, solem, measures, y_test)
    end

    return ModelSet(ds, solem, nothing, measures)
end

function paso_analysis(
    ds::SoleXplorer.EitherDataSet,
    solem::SoleModel;
    kwargs...
)::ModelSet
    _paso_analysis(ds, solem; kwargs...)
end

function paso_analysis(
    X::AbstractDataFrame,
    y::AbstractVector,
    w::SoleXplorer.OptVector = nothing;
    extractor::Union{Nothing,SoleXplorer.RuleExtractor}=nothing,
    measures::Tuple{Vararg{SoleXplorer.FussyMeasure}}=(),
    kwargs...
)::ModelSet
    ds = SoleXplorer._setup_dataset(X, y, w; kwargs...)
    solem = _paso_test(ds)
    _paso_analysis(ds, solem; extractor, measures)
end

paso_dtc = paso_analysis(
    Xc, yc,
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

model_rfc = symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)

model_rfc = paso_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
)