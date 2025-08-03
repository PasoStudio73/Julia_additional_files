# ---------------------------------------------------------------------------- #
#                          preparazione esperimento                            #
# ---------------------------------------------------------------------------- #
using SoleXplorer
using SoleModels
using MLJ
using DataFrames, Random
using Test, BenchmarkTools
import DecisionTree as DT

Xc, yc = @load_iris
Xc = DataFrame(Xc)

# ---------------------------------------------------------------------------- #
#              NUOVE STRUTTURE DecisionEnsemble e DecisionTree                 #
# ---------------------------------------------------------------------------- #
const Branch_or_Leaf{O} = Union{Branch{O}, LeafModel{O}}

default_parity_func = x->mode(x)
dt_parity_func = x->mode(sort(x))

# ---------------------------------------------------------------------------- #
#              NUOVE STRUTTURE DecisionEnsemble e DecisionTree                 #
# ---------------------------------------------------------------------------- #
# The vector featim contains the feature importance scores for each feature in your dataset.
# These scores indicate how much each feature contributes to the decision-making process
# across all trees in the random forest.
struct PasoDecisionEnsemble{O,T<:AbstractModel,A<:Base.Callable} <: SoleModels.AbstractDecisionEnsemble{O}
    models      :: Vector{T}
    aggregation :: A
    weights     :: Union{Nothing,Vector{<:Real}}
    featim      :: Union{Nothing,Vector{Float64}}

    function PasoDecisionEnsemble{O}(
        models      :: Vector{T};
        aggregation :: Base.Callable=default_parity_func,
        weights     :: Union{Nothing,Vector{<:Real}}=nothing,
        featim      :: Union{Nothing,Vector{Float64}}=nothing,
    )::PasoDecisionEnsemble where {O,T<:AbstractModel}
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)

        A = typeof(aggregation)
        new{O,T,A}(collect(models), aggregation, weights, featim)
    end

    function PasoDecisionEnsemble(models::AbstractVector{Any}; kwargs...)::PasoDecisionEnsemble
        @assert length(models) > 0 "Cannot instantiate empty ensemble!"
        models = wrap.(models)
        O = Union{outcometype.(models)...}
        PasoDecisionEnsemble{O}(models; kwargs...)
    end
end

struct PasoDecisionTree{O} <: AbstractModel{O}
    root::Branch_or_Leaf{O}

    function PasoDecisionTree(root::Branch_or_Leaf{O})::PasoDecisionTree where {O}
        new{O}(root)
    end

    function PasoDecisionTree(root::Any)::PasoDecisionTree
        root = wrap(root)
        M    = typeof(root)
        O    = outcometype(root)
        @assert M <: Union{LeafModel{O},Branch{O}} "" *
            "Cannot instantiate PasoDecisionTree{$(O)}(...) with root of " *
            "type $(typeof(root)). Note that the should be either a LeafModel or a " *
            "Branch. " *
            "$(M) <: $(Union{LeafModel,Branch{<:O}}) should hold."
        new{O}(root)
    end

    function PasoDecisionTree(
        antecedent    :: Formula,
        posconsequent :: Any,
        negconsequent :: Any,
    )::PasoDecisionTree
        posconsequent isa PasoDecisionTree && (posconsequent = root(posconsequent))
        negconsequent isa PasoDecisionTree && (negconsequent = root(negconsequent))
        return PasoDecisionTree(Branch(antecedent, posconsequent, negconsequent))
    end
end

# ---------------------------------------------------------------------------- #
#                          NUOVA FUNZIONE solemodel                            #
# ---------------------------------------------------------------------------- #
function get_condition(featid, featval)
    test_operator = (<)
    feature = VariableValue(featid)
    ScalarCondition(feature, test_operator, featval)
end

function pasomodel(
    model       :: DT.Ensemble{T,O};
    aggregation :: Base.Callable=dt_parity_func,
    weights     :: Union{Nothing,Vector{<:Real}}=nothing 
)::PasoDecisionEnsemble where {T,O}
    trees = map(t -> pasomodel(t), model.trees)

    hasproperty(model, :featim) ?
        PasoDecisionEnsemble{O}(trees; aggregation, weights, featim=model.featim) :
        PasoDecisionEnsemble{O}(trees; aggregation, weights)
end

function pasomodel(tree::DT.InfoNode{T,O};)::PasoDecisionTree where {T,O}
    root = pasomodel(tree.node)
    PasoDecisionTree(root)
end

function pasomodel(tree::DT.Node)::Branch
    cond       = get_condition(tree.featid, tree.featval)
    antecedent = Atom(cond)
    lefttree   = pasomodel(tree.left)
    righttree  = pasomodel(tree.right)
    Branch(antecedent, lefttree, righttree)
end

function pasomodel(tree::DT.Leaf)::ConstantModel
    SoleModels.ConstantModel(tree.majority)
end

# ---------------------------------------------------------------------------- #
#                            NUOVA STRUTTURA Info                              #
# ---------------------------------------------------------------------------- #
abstract type AbstractInfo{T,O} end

function get_refs(y::AbstractVector)::Vector{UInt32}
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
    featurenames :: Union{Nothing,Vector{Symbol}}
    classlabels  :: Union{Nothing,Vector{Symbol}}

    # Generic constructor
    function Info{O}(
        labels       :: Vector{T},
        predictions  :: Vector{T};
        featurenames :: Union{Nothing,AbstractVector}=nothing,
        classlabels  :: Union{Nothing,AbstractVector}=nothing
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
        labels      :: AbstractVector,
        predictions :: AbstractVector;
        kwargs...
    )::Info
        labels_refs = get_refs(labels)
        preds_refs  = get_refs(predictions)
        Info{eltype(labels)}(labels_refs, preds_refs; kwargs...)
    end

    # Regression constructor
    function Info(
        labels       :: AbstractVector{L},
        predictions  :: AbstractVector{L};
        float32      :: Bool=False,
        kwargs...
    )::Info where {L<:SoleModels.RLabel}
        float32 ?
            Info{eltype(labels)}(Float32.(labels), Float32.(predictions); kwargs...) :
            Info{eltype(labels)}(labels, predictions; kwargs...)
    end
end

struct SoleModel{D} <: SoleXplorer.AbstractSoleModel
    model::Union{PasoDecisionEnsemble, PasoDecisionTree}
    info::Info

    function SoleModel(::D, model::Union{PasoDecisionEnsemble, PasoDecisionTree}, info::Info) where D<:SoleXplorer.AbstractDataSet
        new{D}(model, info)
    end
end

# ---------------------------------------------------------------------------- #
#                            NUOVA funzione apply!                             #
# ---------------------------------------------------------------------------- #
pasoroot(m::PasoDecisionTree) = m.root

models(m::PasoDecisionEnsemble) = m.models
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

function pasoapply(
    m::PasoDecisionEnsemble,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector{<:SoleModels.CLabel};
    featurenames::AbstractVector,
    classlabels::AbstractVector,
    leavesonly = false,
    kwargs...
)::Info
    preds = hcat([pasoapply(subm, d, y; mode, leavesonly, kwargs...) for subm in models(m)]...)
    preds = map(eachrow(preds)) do row
        weighted_aggregation(m)(row; kwargs...)
    end

    Info(y, preds; featurenames, classlabels)
end

function pasoapply(
    m::PasoDecisionTree,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector{<:SoleModels.CLabel};
    featurenames::AbstractVector,
    classlabels::AbstractVector,
    leavesonly = false,
    kwargs...
)::Info
    preds = pasoapply(pasoroot(m), d, y; leavesonly, kwargs...)

    Info(y, preds; featurenames, classlabels)
end

function pasoapply(
    m::Branch,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    check_args::Tuple = (),
    check_kwargs::NamedTuple = (;),
    leavesonly = false,
    kwargs...
)
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

    return preds
end

function pasoapply(
    m::ConstantModel,
    d::SoleModels.AbstractInterpretationSet,
    y::AbstractVector;
    leavesonly = false,
    kwargs...
)
    return fill(outcome(m), ninstances(d))
end

# ---------------------------------------------------------------------------- #
#                                     Test                                     #
# ---------------------------------------------------------------------------- #
dsc = setup_dataset(
    Xc, yc;
    model=DecisionTreeClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)

rfc = setup_dataset(
    Xc, yc;
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
        train_ratio=0.7,
        rng=Xoshiro(1),   
)

train, test = get_train(dsc.pidxs[1]), get_test(dsc.pidxs[1])
X_test = @views get_X(dsc)[test, :]
y_test = get_y(dsc)[test]
MLJ.fit!(dsc.mach, rows=train, verbosity=0)
MLJ.fit!(rfc.mach, rows=train, verbosity=0)

# SoleXplorer's apply
@test dsc isa SoleXplorer.DecisionTreeApply
@test rfc isa SoleXplorer.PropositionalDataSet{RandomForestClassifier}

featurenames = MLJ.report(dsc.mach).features
classlabels  = report(dsc.mach).classes_seen
tree = MLJ.fitted_params(dsc.mach).tree
forest = MLJ.fitted_params(rfc.mach).forest

@test_nowarn pasomodel(tree);
@test_nowarn pasomodel(forest);

md = pasomodel(tree);
mf = pasomodel(forest);

logiset = scalarlogiset(X_test, allow_propositional = true)

@test_nowarn pasoapply(md, logiset, y_test; featurenames, classlabels);
@test_nowarn pasoapply(mf, logiset, y_test; featurenames, classlabels);

dsi = pasoapply(md, logiset, y_test; featurenames, classlabels);
rfi = pasoapply(mf, logiset, y_test; featurenames, classlabels);

# ---------------------------------------------------------------------------- #
#                        nuova SoleXplorer train_test                          #
# ---------------------------------------------------------------------------- #
function xplorer_apply(
    ds :: SoleXplorer.DecisionTreeApply,
    X  :: AbstractDataFrame,
    y  :: AbstractVector
)
    featurenames = MLJ.report(ds.mach).features
    classlabels  = MLJ.report(ds.mach).classes_seen
    solem        = pasomodel(MLJ.fitted_params(ds.mach).tree)
    logiset      = scalarlogiset(X, allow_propositional = true)
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
        solemodel[i] = SoleModel(ds, solem, info)
    end

    return solemodel
end

function paso_test(args...; kwargs...)::Vector{SoleModel}
    ds = SoleXplorer._setup_dataset(args...; kwargs...)
    _paso_test(ds)
end

paso_test(ds::SoleXplorer.AbstractDataSet)::Vector{SoleModel} = _paso_test(ds)

# ---------------------------------------------------------------------------- #
#                                     Test                                     #
# ---------------------------------------------------------------------------- #
@test_nowarn paso_test(dsc);
@test_nowarn paso_test(rfc);

solemc = paso_test(dsc);
solemc = paso_test(rfc);

# ---------------------------------------------------------------------------- #
#                           NUOVO SYMBOLIC ANALYSIS                            #
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
);

model_rfc = symbolic_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);

paso_rfc = paso_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);

@btime paso_analysis(
    Xc, yc,
    model=RandomForestClassifier(),
    resample=Holdout(shuffle=true),
    train_ratio=0.7,
    rng=Xoshiro(1),
    measures=(accuracy, kappa)
);

