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

const Optional{T} = Union{T, Nothing}
const OptCallable = Optional{Base.Callable}
# const OptReal = Optional{Real}
# const OptAbstractVector{T} = Optional{AbstractVector{T}}

const Branch_or_Leaf{O} = Union{Branch{O}, LeafModel{O}}

const default_parity_func = x->argmax(x)

struct PasoDecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable,W<:Real} <: SoleModels.AbstractDecisionEnsemble{O}
    models::Vector{T}
    aggregation::A
    weights::Vector{W}

    function PasoDecisionEnsemble{O}(
        models::AbstractVector{T},
        aggregation::OptCallable,
        weights::AbstractVector{W};
        suppress_parity_warning::Bool=false,
        parity_func::Base.Callable=default_parity_func
    ) where {O,T<:AbstractModel, W<:Real}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"

        models = wrap.(models)

        if isnothing(aggregation)
            aggregation = function (args...; suppress_parity_warning, kwargs...) SoleModels.bestguess(args...; suppress_parity_warning, parity_func, kwargs...) end
        else
            !suppress_parity_warning || @warn "Unexpected value for suppress_parity_warning: $(suppress_parity_warning)."
        end

        A = typeof(aggregation)

        new{O,T,A,W}(collect(models), aggregation, weights)
    end
    
    function PasoDecisionEnsemble{O}(
        models::AbstractVector;
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, Vector{Float64}(); kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        aggregation::OptCallable;
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, aggregation, Vector{Float64}(); kwargs...)
    end

    function PasoDecisionEnsemble{O}(
        models::AbstractVector,
        weights::AbstractVector;
        kwargs...
    ) where {O}
        PasoDecisionEnsemble{O}(models, nothing, weights; kwargs...)
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
#                         NUOVA STRUTTURA Info                           #
# ---------------------------------------------------------------------------- #
# Questa struttura andrebbe a sostituire la vecchia NamedTuple Info,
# e include la seconda idea di Gio riguardo alla classificazione,
# cioè salvare labels e predictions come refs di categorical.
# Avendo cura di salvarsi le classlabels ordinate, altrimenti non si può tornare indietro.
# Le classlabels, così come le già viste featurenames,
# In questo punto potrebbero servire solo per la print (per ora uso il condizionale)

abstract type AbstractInfo{T,O} end

struct Info{T,O} <: AbstractInfo{T,O}
    labels::Vector{T}
    predictions::Vector{T}
    featurenames::Vector{Symbol}
    classlabels::Vector{Symbol}

    function Info{O}(
        labels::Vector{UInt32},
        predictions::Vector{UInt32};
        featurenames::AbstractVector,
        classlabels::AbstractVector
    ) where {O<:SoleModels.Label}
        fnames = eltype(featurenames) <: Symbol ? featurenames : Symbol.(featurenames)
        clabels = eltype(classlabels) <: Symbol ? classlabels : Symbol.(classlabels)
        sort!(clabels)
        new{UInt32, O}(labels, predictions, fnames, clabels)
    end

    # Classification constructor
    function Info(
        labels::AbstractVector{L},
        predictions::AbstractVector{L};
        kwargs...
    ) where {L<:SoleModels.CLabel}
        label_levels = labels isa MLJ.CategoricalArray ? labels : categorical(labels, levels=sort(unique(labels)))
        preds_levels = predictions isa MLJ.CategoricalArray ? predictions : categorical(predictions, levels=sort(unique(predictions)))
        Info{eltype(labels)}(label_levels.refs, preds_levels.refs; kwargs...)
    end

    # Regression constructor
    function Info(
        labels::AbstractVector{L},
        predictions::AbstractVector{L};
        featurenames::AbstractVector,
        float32::Bool=False
    ) where {L<:SoleModels.RLabel}
        fnames = eltype(featurenames) <: Symbol ? featurenames : Symbol.(featurenames)
        float32 ?
            new{Float32, eltype(labels)}(Float32.(labels), Float32.(predictions), fnames, Symbol()) :
            new{eltype(labels), eltype(labels)}(labels, predictions, fnames, Symbol())
    end
end



